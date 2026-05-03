// ═══════════════════════════════════════════════════════════════════════════
//  Spiral Shooter  ·  PPO  ·  Unified network  ·  No deadlock
// ═══════════════════════════════════════════════════════════════════════════
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <numeric>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

// ────────────────────────────────────────────────────────
//  Tunables
// ────────────────────────────────────────────────────────
static constexpr int   N_ENVS       = 24;
static constexpr int   TRAIN_EVERY  = 4096;
static constexpr int   PPO_EPOCHS   = 4;
static constexpr int   BATCH_SIZE   = 512;
static constexpr int   MAX_BUF      = 16384;
static constexpr int   WIDTH        = 1600;
static constexpr int   HEIGHT       = 900;

static constexpr float GAMMA        = 0.99f;
static constexpr float LAMBDA_GAE   = 0.95f;
static constexpr float PPO_CLIP     = 0.2f;
static constexpr float LR_ACTOR     = 2e-4f;
static constexpr float LR_CRITIC    = 5e-4f;
static constexpr float ENT_CONT     = 0.003f;
static constexpr float ENT_BIN      = 0.01f;

static constexpr float EPS_START    = 0.5f;
static constexpr float EPS_MIN      = 0.12f;
static constexpr float EPS_DECAY    = 0.995f;

// Aiming gate — bullet only fires if lead_align exceeds this
static constexpr float SHOOT_GATE   = 0.4f;
static constexpr float BULLET_SPD   = 11.f;

// Network dims
static constexpr int   MAX_E_TRACK  = 3;
static constexpr int   MAX_B_TRACK  = 3;
static constexpr int   S_DIM        = 4 + (MAX_E_TRACK * 9) + (MAX_B_TRACK * 5) + 2; // 48
static constexpr int   STACK        = 3;
static constexpr int   IN_DIM       = S_DIM * STACK;
static constexpr int   H1 = 256, H2 = 256, H3 = 128;
static constexpr int   A_MOVE_DIM   = 2;   // [move_x, move_y]
static constexpr int   A_AIM_DIM    = 3;   // [aim_sin, aim_cos, shoot]

enum { A_MX=0, A_MY };
enum { A_SIN=0, A_COS, A_SH };

// ────────────────────────────────────────────────────────
//  Math
// ────────────────────────────────────────────────────────
struct V2 {
    float x=0,y=0;
    V2()=default; V2(float x,float y):x(x),y(y){}
    V2 operator+(V2 b)const{return{x+b.x,y+b.y};}
    V2 operator-(V2 b)const{return{x-b.x,y-b.y};}
    V2 operator*(float s)const{return{x*s,y*s};}
    V2& operator+=(V2 b){x+=b.x;y+=b.y;return*this;}
    float dot(V2 b)const{return x*b.x+y*b.y;}
    float len()const{return std::sqrt(x*x+y*y);}
    float dist(V2 b)const{return (*this-b).len();}
    V2 norm()const{float l=len();return l>1e-6f?V2{x/l,y/l}:V2{0,0};}
};

static V2 predict_target(V2 p_pos, V2 e_pos, V2 e_vel, float b_speed) {
    V2 pred = e_pos;
    for (int i = 0; i < 3; i++) {
        float t = p_pos.dist(pred) / b_speed;
        pred = e_pos + e_vel * t;
    }
    return pred;
}

// ────────────────────────────────────────────────────────
//  Per-thread RNG
// ────────────────────────────────────────────────────────
static thread_local std::mt19937 g_rng{std::random_device{}()};
namespace RNG {
    float flt(float a,float b){return std::uniform_real_distribution<float>(a,b)(g_rng);}
    int   rint(int a,int b)   {return std::uniform_int_distribution<int>(a,b)(g_rng);}
    float norm(float m=0,float s=1){return std::normal_distribution<float>(m,s)(g_rng);}
}

// ────────────────────────────────────────────────────────
//  Linear algebra
// ────────────────────────────────────────────────────────
using Mat = std::vector<std::vector<float>>;
using Vec = std::vector<float>;

Mat mmat(int r,int c,float v=0){return Mat(r,Vec(c,v));}
Vec mvmul(const Mat&W,const Vec&x){
    Vec o(W.size(),0);
    for(int i=0;i<(int)W.size();i++) for(int j=0;j<(int)x.size();j++) o[i]+=W[i][j]*x[j];
    return o;
}
void vadd(Vec&v,const Vec&b){for(int i=0;i<(int)v.size();i++)v[i]+=b[i];}
Vec lrelu(const Vec&v,float a=0.01f){Vec o(v.size()); for(int i=0;i<(int)v.size();i++) o[i]=v[i]>=0?v[i]:a*v[i]; return o;}
Vec dlrelu(const Vec&p,float a=0.01f){Vec o(p.size()); for(int i=0;i<(int)p.size();i++) o[i]=p[i]>=0?1.f:a; return o;}
static void clip_grad(Mat&dW,Vec&db,float mx=0.5f){
    float n=0; for(auto&r:dW) for(float g:r) n+=g*g; for(float g:db) n+=g*g;
    n=std::sqrt(n); if(n>mx){float s=mx/(n+1e-8f); for(auto&r:dW) for(float&g:r) g*=s; for(float&g:db) g*=s;}
}

// ────────────────────────────────────────────────────────
//  Adam
// ────────────────────────────────────────────────────────
struct Adam {
    Mat mW,vW; Vec mb,vb; int t=0;
    Adam()=default;
    Adam(int r,int c):mW(mmat(r,c)),vW(mmat(r,c)),mb(r,0),vb(r,0){}
    void step(Mat&W,Vec&b,const Mat&dW,const Vec&db,float lr,float wd=1e-4f,
              float b1=0.9f,float b2=0.999f,float ep=1e-8f){
        ++t; float c1=1-std::pow(b1,t),c2=1-std::pow(b2,t);
        for(int i=0;i<(int)W.size();i++){
            for(int j=0;j<(int)W[0].size();j++){
                W[i][j]*=(1-wd);
                mW[i][j]=b1*mW[i][j]+(1-b1)*dW[i][j];
                vW[i][j]=b2*vW[i][j]+(1-b2)*dW[i][j]*dW[i][j];
                W[i][j]+=lr*(mW[i][j]/c1)/(std::sqrt(vW[i][j]/c2)+ep);
            }
            mb[i]=b1*mb[i]+(1-b1)*db[i]; vb[i]=b2*vb[i]+(1-b2)*db[i]*db[i];
            b[i]+=lr*(mb[i]/c1)/(std::sqrt(vb[i]/c2)+ep);
        }
    }
};

// ────────────────────────────────────────────────────────
//  Neural Network (Base for Move and Aim)
// ────────────────────────────────────────────────────────
struct Net {
    int a_dim;
    Mat w1,w2,w3,wa,wc;
    Vec b1,b2,b3,ba,bc;
    Mat ow1,ow2,ow3,owa;
    Vec ob1,ob2,ob3,oba;
    Adam a1,a2,a3,aa,ac;
    float lr_a,lr_c;

    struct Cache{ Vec h1,h2,h3,p1,p2,p3; };

    Net(int adim, float la=LR_ACTOR,float lc=LR_CRITIC):a_dim(adim), lr_a(la),lr_c(lc){
        auto he=[](int fi){return RNG::norm(0,std::sqrt(2.f/fi));};
        auto init=[&](Mat&W,Vec&b,int r,int c){
            W=mmat(r,c); b.assign(r,0);
            for(auto&row:W) for(auto&w:row) w=he(c);
        };
        init(w1,b1,H1,IN_DIM); init(w2,b2,H2,H1); init(w3,b3,H3,H2);
        init(wa,ba,a_dim,H3);  init(wc,bc,1,H3);
        for(auto&r:wa) for(auto&w:r) w*=0.01f;
        for(auto&r:wc) for(auto&w:r) w*=0.01f;
        if(a_dim == A_AIM_DIM) ba[A_SH]=-3.f;  // start not shooting
        snapshot();
        a1=Adam(H1,IN_DIM); a2=Adam(H2,H1); a3=Adam(H3,H2);
        aa=Adam(a_dim,H3);  ac=Adam(1,H3);
    }

    std::pair<Vec,float> fwd(const Vec&s,Cache*c=nullptr)const{
        Vec p1=mvmul(w1,s); vadd(p1,b1); Vec h1=lrelu(p1);
        Vec p2=mvmul(w2,h1);vadd(p2,b2); Vec h2=lrelu(p2);
        Vec p3=mvmul(w3,h2);vadd(p3,b3); Vec h3=lrelu(p3);
        Vec ao=mvmul(wa,h3); vadd(ao,ba);
        Vec co=mvmul(wc,h3); vadd(co,bc);
        if(a_dim == A_MOVE_DIM) {
            for(int i=0;i<2;i++) ao[i]=std::tanh(ao[i]);
        } else {
            ao[A_SIN]=std::tanh(ao[A_SIN]);
            ao[A_COS]=std::tanh(ao[A_COS]);
            ao[A_SH]=1.f/(1+std::exp(-ao[A_SH]));
        }
        if(c)*c={h1,h2,h3,p1,p2,p3};
        return{ao,co[0]};
    }

    Vec fwd_old(const Vec&s)const{
        Vec p1=mvmul(ow1,s); vadd(p1,ob1); Vec h1=lrelu(p1);
        Vec p2=mvmul(ow2,h1);vadd(p2,ob2); Vec h2=lrelu(p2);
        Vec p3=mvmul(ow3,h2);vadd(p3,ob3); Vec h3=lrelu(p3);
        Vec ao=mvmul(owa,h3); vadd(ao,oba);
        if(a_dim == A_MOVE_DIM) {
            for(int i=0;i<2;i++) ao[i]=std::tanh(ao[i]);
        } else {
            ao[A_SIN]=std::tanh(ao[A_SIN]);
            ao[A_COS]=std::tanh(ao[A_COS]);
            ao[A_SH]=1.f/(1+std::exp(-ao[A_SH]));
        }
        return ao;
    }

    void snapshot(){ow1=w1;ow2=w2;ow3=w3;owa=wa;ob1=b1;ob2=b2;ob3=b3;oba=ba;}

    void copy_from(const Net&o){
        w1=o.w1;w2=o.w2;w3=o.w3;wa=o.wa;wc=o.wc;
        b1=o.b1;b2=o.b2;b3=o.b3;ba=o.ba;bc=o.bc;
    }

    void ppo_update(const std::vector<Vec>&states,
                    const std::vector<Vec>&actions,
                    const std::vector<float>&advantages,
                    const std::vector<float>&returns,
                    const std::vector<float>&epsilons,
                    std::mutex&mtx)
    {
        int N=(int)states.size();
        std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
        std::shuffle(idx.begin(),idx.end(),g_rng);

        for(int bs=0;bs<N;bs+=BATCH_SIZE){
            int be=std::min(bs+BATCH_SIZE,N);
            Mat dw1=mmat(H1,IN_DIM),dw2=mmat(H2,H1),dw3=mmat(H3,H2);
            Mat dwa=mmat(a_dim,H3),dwc=mmat(1,H3);
            Vec db1(H1,0),db2(H2,0),db3(H3,0),dba(a_dim,0),dbc(1,0);
            int used=0;

            for(int bi=bs;bi<be;bi++){
                int i=idx[bi];
                float A=advantages[i];
                float eps_i=std::max(epsilons[i],0.05f);

                Cache c;
                auto[cur_act,cur_val]=fwd(states[i],&c);
                Vec old_act=fwd_old(states[i]);

                float log_new=0,log_old=0;
                for(int k=0;k<a_dim;k++){
                    float pn=cur_act[k],po=old_act[k],act=actions[i][k];
                    if(a_dim == A_AIM_DIM && k == A_SH){
                        log_new+=act*std::log(pn+1e-8f)+(1-act)*std::log(1-pn+1e-8f);
                        log_old+=act*std::log(po+1e-8f)+(1-act)*std::log(1-po+1e-8f);
                    } else {
                        float sig=eps_i; if(a_dim == A_AIM_DIM && k<A_SH)sig*=1.5f; sig=std::max(sig,0.05f);
                        log_new+=-0.5f*std::pow((act-pn)/sig,2);
                        log_old+=-0.5f*std::pow((act-po)/sig,2);
                    }
                }
                float ratio=std::exp(std::clamp(log_new-log_old,-5.f,5.f));
                float clip_r=std::clamp(ratio,1-PPO_CLIP,1+PPO_CLIP);
                bool clipped=(A>0&&ratio>1+PPO_CLIP)||(A<0&&ratio<1-PPO_CLIP);
                float eff=clipped?clip_r:ratio;

                Vec oa(a_dim,0);
                for(int k=0;k<a_dim;k++){
                    float pn=cur_act[k],act=actions[i][k];
                    if(a_dim == A_AIM_DIM && k == A_SH){
                        oa[k]=eff*A*(act-pn)+ENT_BIN*(0.5f-pn);
                    } else {
                        float sig=eps_i; if(a_dim == A_AIM_DIM && k<A_SH)sig*=1.5f; sig=std::max(sig,0.05f);
                        float dt=std::max(1e-3f,1-pn*pn);
                        oa[k]=eff*A*(act-pn)/(sig*sig)*dt-ENT_CONT*pn*dt;
                    }
                }
                float cerr=std::clamp(returns[i]-cur_val,-5.f,5.f);

                Vec h3e(H3,0);
                for(int j=0;j<a_dim;j++){
                    dba[j]+=oa[j];
                    for(int ii=0;ii<H3;ii++){dwa[j][ii]+=oa[j]*c.h3[ii]; h3e[ii]+=wa[j][ii]*oa[j];}
                }
                dbc[0]+=cerr;
                for(int ii=0;ii<H3;ii++){dwc[0][ii]+=cerr*c.h3[ii]; h3e[ii]+=wc[0][ii]*cerr;}

                Vec d3=dlrelu(c.p3); Vec h2e(H2,0);
                for(int j=0;j<H3;j++){
                    float g=h3e[j]*d3[j]; db3[j]+=g;
                    for(int ii=0;ii<H2;ii++){dw3[j][ii]+=g*c.h2[ii]; h2e[ii]+=w3[j][ii]*g;}
                }
                Vec d2=dlrelu(c.p2); Vec h1e(H1,0);
                for(int j=0;j<H2;j++){
                    float g=h2e[j]*d2[j]; db2[j]+=g;
                    for(int ii=0;ii<H1;ii++){dw2[j][ii]+=g*c.h1[ii]; h1e[ii]+=w2[j][ii]*g;}
                }
                Vec d1=dlrelu(c.p1);
                for(int j=0;j<H1;j++){
                    float g=h1e[j]*d1[j]; db1[j]+=g;
                    for(int ii=0;ii<IN_DIM;ii++) dw1[j][ii]+=g*states[i][ii];
                }
                used++;
            }
            if(!used)continue;
            float sc=1.f/used;
            auto norm=[&](Mat&M,Vec&v){for(auto&r:M)for(auto&x:r)x*=sc;for(auto&x:v)x*=sc;};
            norm(dw1,db1);norm(dw2,db2);norm(dw3,db3);norm(dwa,dba);norm(dwc,dbc);
            clip_grad(dw1,db1);clip_grad(dw2,db2);clip_grad(dw3,db3);clip_grad(dwa,dba);clip_grad(dwc,dbc);

            std::lock_guard<std::mutex> lk(mtx);
            a1.step(w1,b1,dw1,db1,lr_a); a2.step(w2,b2,dw2,db2,lr_a);
            a3.step(w3,b3,dw3,db3,lr_a); aa.step(wa,ba,dwa,dba,lr_a);
            ac.step(wc,bc,dwc,dbc,lr_c);
        }
    }

    void save(const std::string&p)const{
        std::ofstream f(p,std::ios::binary); if(!f)return;
        auto wm=[&](const Mat&M){for(auto&r:M)f.write((char*)r.data(),r.size()*4);};
        auto wv=[&](const Vec&v){f.write((char*)v.data(),v.size()*4);};
        wm(w1);wm(w2);wm(w3);wm(wa);wm(wc);wv(b1);wv(b2);wv(b3);wv(ba);wv(bc);
    }
    void load(const std::string&p){
        std::ifstream f(p,std::ios::binary); if(!f){std::cout<<"[Net] Fresh start\n";return;}
        auto rm=[&](Mat&M){for(auto&r:M)f.read((char*)r.data(),r.size()*4);};
        auto rv=[&](Vec&v){f.read((char*)v.data(),v.size()*4);};
        rm(w1);rm(w2);rm(w3);rm(wa);rm(wc);rv(b1);rv(b2);rv(b3);rv(ba);rv(bc);
        snapshot(); std::cout<<"[Net] Loaded "<<p<<"\n";
    }
};

// ────────────────────────────────────────────────────────
//  Experience buffer
// ────────────────────────────────────────────────────────
struct Exp{Vec s, am, aa, ns; float rm, ra; bool done; float eps;};
using Episode = std::vector<Exp>;
class Buf{
    std::vector<Episode> d; std::mutex m;
public:
    void push_episode(Episode ep){
        std::lock_guard lk(m); d.push_back(std::move(ep));
        if((int)d.size()>500) d.erase(d.begin(), d.begin()+(d.size()-500));
    }
    std::vector<Episode> swap_out(){std::lock_guard lk(m);std::vector<Episode> o;o.swap(d);return o;}
    int size(){std::lock_guard lk(m);return(int)d.size();}
};

// ────────────────────────────────────────────────────────
//  Game entities
// ────────────────────────────────────────────────────────
struct Bullet{
    V2 pos,vel; int sz; SDL_Color col; bool alive=true,hit=false;
    Bullet(V2 p,V2 t,SDL_Color c,int sz,float spd):pos(p),sz(sz),col(c){vel=(t-p).norm()*spd;}
    void update(){pos+=vel;if(pos.x<-60||pos.x>WIDTH+60||pos.y<-60||pos.y>HEIGHT+60)alive=false;}
    SDL_Rect rect()const{return{(int)(pos.x-sz/2),(int)(pos.y-sz/2),sz,sz};}
    void draw(SDL_Renderer*r,int ox,int oy){
        SDL_Rect rc={(int)(pos.x-sz/2)+ox,(int)(pos.y-sz/2)+oy,sz,sz};
        SDL_SetRenderDrawColor(r,col.r,col.g,col.b,col.a);SDL_RenderFillRect(r,&rc);
    }
};

struct Particle{
    V2 pos,vel; int life=28; SDL_Color col;
    Particle(V2 p,SDL_Color c):pos(p),vel{RNG::flt(-3,3),RNG::flt(-3,3)},col(c){}
    void update(){pos+=vel;life--;}
    bool alive()const{return life>0;}
    void draw(SDL_Renderer*r,int ox,int oy){
        SDL_SetRenderDrawColor(r,col.r,col.g,col.b,255);
        SDL_Rect rc={(int)pos.x+ox,(int)pos.y+oy,4,4};SDL_RenderFillRect(r,&rc);
    }
};

struct Player{
    V2 pos{WIDTH/2.f,HEIGHT/2.f},vel;
    int hp=100,cd=0; SDL_Texture*tex=nullptr;
    void update(){if(cd>0)cd--;}
    void move(V2 dir,bool restrict){
        if(dir.len()>1)dir=dir.norm();
        vel=dir*6.f; pos+=vel;
        auto cl=[](float v,float a,float b){return std::max(a,std::min(b,v));};
        if(restrict){pos.x=cl(pos.x,WIDTH/2-295,WIDTH/2+295);pos.y=cl(pos.y,HEIGHT/2-295,HEIGHT/2+295);}
        else{pos.x=cl(pos.x,15,(float)WIDTH-15);pos.y=cl(pos.y,15,(float)HEIGHT-15);}
    }
    bool can_shoot()const{return cd==0;}
    void fire(){cd=8;}
    void hurt(int d){hp-=d;}
    SDL_Rect rect()const{return{(int)(pos.x-14),(int)(pos.y-14),28,28};}
    void draw(SDL_Renderer*r,int ox,int oy){
        SDL_Rect rc={(int)(pos.x-14)+ox,(int)(pos.y-14)+oy,28,28};
        if(tex)SDL_RenderCopyEx(r,tex,nullptr,&rc,0,nullptr,SDL_FLIP_NONE);
        else{SDL_SetRenderDrawColor(r,180,210,255,255);SDL_RenderFillRect(r,&rc);}
        SDL_SetRenderDrawColor(r,255,255,100,255);SDL_RenderDrawRect(r,&rc);
    }
};

struct Enemy{
    V2 pos,vel_smooth,prev_pos;
    float spiral=0; int hp=150,cd; bool alive=true;
    SDL_Texture*tex=nullptr;
    Enemy(){
        int s=RNG::rint(0,3);
        if(s==0)pos={RNG::flt(0,WIDTH),-35};
        else if(s==1)pos={(float)WIDTH+35,RNG::flt(0,HEIGHT)};
        else if(s==2)pos={RNG::flt(0,WIDTH),(float)HEIGHT+35};
        else pos={-35,RNG::flt(0,HEIGHT)};
        spiral=RNG::flt(1.5f,4.5f)*(RNG::rint(0,1)?1:-1)/100.f;
        cd=RNG::rint(10,25);prev_pos=pos;
    }
    void update(Player&p,std::vector<Bullet>&eb){
        V2 to=p.pos-pos; float d=to.len(); V2 dir=to.norm();
        if(d>200)pos+=dir*2.4f;
        V2 perp={-dir.y,dir.x}; pos+=perp*(spiral*200);
        if(RNG::rint(0,120)==0)spiral*=-1;
        if(std::abs(spiral)>0.06f)spiral*=0.95f;
        pos.x=std::clamp(pos.x,15.f,(float)WIDTH-15);
        pos.y=std::clamp(pos.y,15.f,(float)HEIGHT-15);
        V2 raw=pos-prev_pos; vel_smooth=vel_smooth*0.7f+raw*0.3f; prev_pos=pos;
        if(--cd<=0){
            float bspd=8.0f,t=d/bspd;
            V2 noise{RNG::flt(-6,6),RNG::flt(-6,6)};
            V2 pred=p.pos+p.vel*t*0.6f+noise;
            eb.push_back(Bullet(pos,pred,{255,120,120,255},12,bspd));
            cd=RNG::rint(60,180);
        }
    }
    void hurt(int d){hp-=d;if(hp<=0)alive=false;}
    SDL_Rect rect()const{return{(int)(pos.x-14),(int)(pos.y-14),28,28};}
    void draw(SDL_Renderer*r,int ox,int oy){
        SDL_Rect rc={(int)(pos.x-14)+ox,(int)(pos.y-14)+oy,28,28};
        if(tex)SDL_RenderCopyEx(r,tex,nullptr,&rc,0,nullptr,SDL_FLIP_NONE);
        else{SDL_SetRenderDrawColor(r,255,80,80,255);SDL_RenderFillRect(r,&rc);}
        SDL_SetRenderDrawColor(r,255,200,0,255);SDL_RenderDrawRect(r,&rc);
    }
};

// ────────────────────────────────────────────────────────
//  State encoder  (S_DIM = 48)
// ────────────────────────────────────────────────────────
Vec encode(const Player&p,
           const std::vector<Enemy>&enemies,
           const std::vector<Bullet>&e_buls,
           int sf,int sh)
{
    Vec s; s.reserve(S_DIM);
    // 1. Player (4)
    s.push_back((p.pos.x - WIDTH/2.f)/(WIDTH/2.f)); 
    s.push_back((p.pos.y - HEIGHT/2.f)/(HEIGHT/2.f));
    s.push_back(p.vel.x/8.f);
    s.push_back(p.vel.y/8.f);

    // 2. Enemies (MAX_E_TRACK * 9 = 27)
    struct EDist { float d; const Enemy* e; };
    std::vector<EDist> ed;
    for(auto&e : enemies) ed.push_back({p.pos.dist(e.pos), &e});
    std::sort(ed.begin(), ed.end(), [](const EDist& a, const EDist& b){ return a.d < b.d; });

    for(int i=0; i<MAX_E_TRACK; i++){
        if(i < (int)ed.size()){
            const Enemy* ne = ed[i].e;
            float d = ed[i].d;
            V2 dir = (ne->pos - p.pos).norm();
            s.push_back(dir.x); s.push_back(dir.y);
            s.push_back(std::min(1.f, d/1200.f));
            s.push_back(ne->vel_smooth.x/5.f); s.push_back(ne->vel_smooth.y/5.f);
            float t = d/BULLET_SPD; V2 lead = ne->pos + ne->vel_smooth*t;
            V2 ld = (lead - p.pos).norm();
            s.push_back(ld.x); s.push_back(ld.y);
            s.push_back(ne->hp/150.f);
            s.push_back(std::min(1.f, (lead - p.pos).len()/1200.f));
        } else {
            for(int j=0; j<9; j++) s.push_back(0.f);
        }
    }

    // 3. Bullets (MAX_B_TRACK * 5 = 15)
    struct BDist { float d; const Bullet* b; };
    std::vector<BDist> bd;
    for(auto&b : e_buls) bd.push_back({p.pos.dist(b.pos), &b});
    std::sort(bd.begin(), bd.end(), [](const BDist& a, const BDist& b){ return a.d < b.d; });

    for(int i=0; i<MAX_B_TRACK; i++){
        if(i < (int)bd.size()){
            const Bullet* nb = bd[i].b;
            V2 dvec = (nb->pos - p.pos).norm(); V2 vn = nb->vel.norm();
            s.push_back(dvec.x); s.push_back(dvec.y);
            s.push_back(std::min(1.f, bd[i].d/600.f));
            s.push_back(vn.x); s.push_back(vn.y);
        } else {
            for(int j=0; j<5; j++) s.push_back(0.f);
        }
    }

    // 4. Global (2)
    s.push_back(p.hp/100.f);
    s.push_back(sf>0?(float)sh/(sf+1):0);

    assert((int)s.size()==S_DIM);
    return s;
}

// ────────────────────────────────────────────────────────
//  Shared sim step
// ────────────────────────────────────────────────────────
struct SimResult{float r;bool done;int hits;};
SimResult sim_step(Player&player,std::vector<Enemy>&enemies,
                   std::vector<Bullet>&pb,std::vector<Bullet>&eb,int&score)
{
    float r=0; bool dead=false; int hits=0;
    SDL_Rect pr=player.rect();

    for(auto it=eb.begin();it!=eb.end();){
        it->update(); SDL_Rect br=it->rect();
        if(SDL_HasIntersection(&br,&pr)){
            player.hurt(40); r-=15.f;
            if(player.hp<=0){r-=20.f;dead=true;}
            it=eb.erase(it);
        } else if(!it->alive)it=eb.erase(it); else ++it;
    }
    if(dead)return{r,true,0};

    for(auto eit=enemies.begin();eit!=enemies.end();){
        eit->update(player,eb);
        SDL_Rect er=eit->rect();
        if(SDL_HasIntersection(&er,&pr)){r-=20.f;return{r,true,0};}
        bool killed=false;
        for(auto bit=pb.begin();bit!=pb.end();){
            SDL_Rect br=bit->rect();
            if(SDL_HasIntersection(&er,&br)){
                eit->hurt(50); r+=80.f; hits++; bit->hit=true;
                bit=pb.erase(bit);
                if(!eit->alive&&!killed){score+=100;r+=60.f;killed=true;}
            } else ++bit;
        }
        if(killed){eit=enemies.erase(eit);continue;}
        ++eit;
    }
    for(auto bit=pb.begin();bit!=pb.end();){
        bit->update();
        if(!bit->alive){if(!bit->hit)r-=8.f; bit=pb.erase(bit);}
        else ++bit;
    }
    return{r,false,hits};
}

// ────────────────────────────────────────────────────────
//  Move shaping reward
// ────────────────────────────────────────────────────────
float move_rew(const Player&p,const std::vector<Enemy>&enemies,const std::vector<Bullet>&eb, V2 move_input)
{
    float r=0;
    float wx=std::min(p.pos.x,(float)WIDTH-p.pos.x);
    float wy=std::min(p.pos.y,(float)HEIGHT-p.pos.y);
    float wall=std::min(wx,wy);
    if(wall<120)r-=2.0f*(1.f-wall/120.f);

    float dc=p.pos.dist({WIDTH/2.f,HEIGHT/2.f});
    r+=0.08f*std::max(0.f,1.f-dc/450.f);  // gentler center pull

    // Anti-camping: penalty for low movement magnitude
    if(move_input.len() < 0.15f) r -= 0.25f;
    else r += 0.05f;

    // Dodge: reward increasing distance from nearest enemy bullet
    float mbd=1e9f;
    for(auto&b:eb){float d=p.pos.dist(b.pos);if(d<mbd)mbd=d;}
    if(mbd<350)r+=0.35f*(mbd/350.f);

    for(auto&e:enemies) if(p.pos.dist(e.pos)<120)r-=1.0f;

    r+=0.02f;
    return r;
}

// ────────────────────────────────────────────────────────
//  Headless environment
// ────────────────────────────────────────────────────────
struct Env{
    Net *shared_move, *shared_aim;
    Net *local_move=nullptr, *local_aim=nullptr;
    std::mutex *net_mtx; Buf*buf;
    std::atomic<float>*eps_ptr;
    int id=0; bool running=true;

    Player player;
    std::vector<Enemy>  enemies;
    std::vector<Bullet> pb,eb;
    std::deque<Vec>     hist;
    std::vector<Exp>    curr_ep;
    int score=0,spawn_t=0,sf=0,sh=0;
    float tot_r=0,last_lead=0;

    void reset(){
        enemies.clear();pb.clear();eb.clear();hist.clear();curr_ep.clear();
        player=Player();score=0;spawn_t=0;sf=0;sh=0;tot_r=0;last_lead=0;
    }

    Vec stacked(){
        Vec cur=encode(player,enemies,eb,sf,sh);
        while((int)hist.size()<STACK)hist.push_back(cur);
        hist.push_back(cur);
        if((int)hist.size()>STACK)hist.pop_front();
        Vec st; for(auto&h:hist)st.insert(st.end(),h.begin(),h.end());
        return st;
    }

    void run(){
        reset(); int sc=0;
        while(running){
            if(sc%128==0){
                std::lock_guard lk(*net_mtx);
                local_move->copy_from(*shared_move);
                local_aim->copy_from(*shared_aim);
            }
            while(running&&buf->size()>200)
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            step(); sc++;
        }
    }

    void step(){
        player.update();
        Vec state=stacked();
        float eps=eps_ptr->load(std::memory_order_relaxed);
        
        auto[act_m, val_m]=local_move->fwd(state);
        auto[act_a, val_a]=local_aim->fwd(state);

        for(int i=0;i<2;i++){act_m[i]+=RNG::norm(0,eps);act_m[i]=std::clamp(act_m[i],-1.f,1.f);}
        for(int i=0;i<2;i++){act_a[i]+=RNG::norm(0,eps*1.5f);act_a[i]=std::clamp(act_a[i],-1.f,1.f);}
        float sp=std::clamp(act_a[A_SH],0.01f,0.99f);
        act_a[A_SH]=(RNG::flt(0,1)<sp)?1.f:0.f;

        player.move({act_m[A_MX],act_m[A_MY]},true);

        float ar=0, mr=0;
        V2 aim{act_a[A_COS],act_a[A_SIN]}; if(aim.len()>1e-5f)aim=aim.norm();
        float lead_a=0;
        const Enemy*nearest=nullptr; float med=1e9f;
        for(auto&e:enemies){float d=player.pos.dist(e.pos);if(d<med){med=d;nearest=&e;}}

        if(nearest){
            V2 pred = predict_target(player.pos, nearest->pos, nearest->vel_smooth, BULLET_SPD);
            lead_a = aim.dot((pred - player.pos).norm());
            
            // Tiered Aim Quality Reward
            if(lead_a > 0.95f)      ar += 2.5f;
            else if(lead_a > 0.85f) ar += 1.2f;
            else if(lead_a > 0.7f)  ar += 0.3f;
            else                    ar -= 0.8f;

            if(act_a[A_SH]>0.5f&&player.can_shoot()){
                ar -= 0.2f; // Ammo cost
                if(lead_a<SHOOT_GATE){act_a[A_SH]=0.f; ar -= 0.5f;} // Extra penalty for spray
            }
        } else { 
            if(act_a[A_SH]>0.5f){act_a[A_SH]=0.f; ar-=1.0f;} // Penalize shooting at ghosts
        }
        last_lead=lead_a;

        if(act_a[A_SH]>0.5f&&player.can_shoot()){
            pb.push_back(Bullet(player.pos,player.pos+aim*1200,{120,255,120,255},10,BULLET_SPD));
            player.fire();sf++;
        }

        mr += move_rew(player,enemies,eb, {act_m[A_MX], act_m[A_MY]});
        SimResult sr=sim_step(player,enemies,pb,eb,score);
        mr+=sr.r; ar+=sr.r;
        sh+=sr.hits;
        if(sr.hits>0)ar+=25.f;

        Vec ns=stacked();
        curr_ep.push_back({state,act_m,act_a,ns,mr,ar,sr.done,eps});
        tot_r+=(mr+ar);

        int max_e=std::min(4,1+score/1000);
        if(++spawn_t>=std::max(90,150-score/300)&&(int)enemies.size()<max_e){enemies.emplace_back();spawn_t=0;}

        if(sr.done){
            float acc=sf>0?100.f*sh/sf:0;
            std::cout<<"[Env "<<id<<"] Sc="<<score<<" Acc="<<acc<<"% R="<<tot_r<<" Shots="<<sf<<"\n";
            buf->push_episode(std::move(curr_ep));
            reset();
        }
    }
};

// ────────────────────────────────────────────────────────
//  Background
// ────────────────────────────────────────────────────────
struct Background{
    SDL_Texture*tex=nullptr;
    Background(SDL_Renderer*r){
        if(!r)return;
        tex=IMG_LoadTexture(r,"back.png");
        if(!tex)tex=IMG_LoadTexture(r,"back.jpg");
        if(tex)SDL_SetTextureColorMod(tex,140,140,190);
    }
    ~Background(){if(tex)SDL_DestroyTexture(tex);}
    void draw(SDL_Renderer*r,int ox,int oy){if(!r||!tex)return;SDL_Rect d={ox,oy,WIDTH,HEIGHT};SDL_RenderCopy(r,tex,nullptr,&d);}
};

// ────────────────────────────────────────────────────────
//  Main Game
// ────────────────────────────────────────────────────────
class Game{
    SDL_Window*win=nullptr; SDL_Renderer*ren=nullptr;
    TTF_Font*font=nullptr; Background*bg=nullptr;
    SDL_Texture*ptex=nullptr,*etex=nullptr;

    Net *mnet=nullptr, *anet=nullptr;
    Net local_mnet{A_MOVE_DIM}, local_anet{A_AIM_DIM};
    Buf buf;
    std::mutex net_mtx;
    std::thread*train_th=nullptr;
    std::vector<Env*>envs;
    std::vector<std::thread*>env_ths;
    std::atomic<float>eps{EPS_START};
    std::atomic<int>ppo_count{0};

    Player player;
    std::vector<Enemy>enemies;
    std::vector<Bullet>pb,eb;
    std::vector<Particle>particles;
    std::deque<Vec>render_hist;
    std::vector<Exp>render_curr_ep;

    int score=0,gen=1,spawn_t=0,shake=0,sf=0,sh=0;
    float tot_r=0,last_lead=0;
    bool running=true,ai=true,headless=false;
    uint8_t keys[SDL_NUM_SCANCODES]={};

public:
    ~Game(){cleanup();}

    bool init(bool a,bool h){
        ai=a;headless=h;
        if(!headless){
            SDL_Init(SDL_INIT_VIDEO);IMG_Init(IMG_INIT_JPG|IMG_INIT_PNG);TTF_Init();
            win=SDL_CreateWindow("Spiral Shooter AI",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,WIDTH,HEIGHT,0);
            ren=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
            const char*fts[]={"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",nullptr};
            for(int i=0;fts[i];i++){font=TTF_OpenFont(fts[i],22);if(font)break;}
            ptex=IMG_LoadTexture(ren,"Player.png");
            etex=IMG_LoadTexture(ren,"Enemy.png");
            bg=new Background(ren);
        } else SDL_Init(0);

        mnet=new Net(A_MOVE_DIM);
        anet=new Net(A_AIM_DIM);
        if(ai){
            mnet->load("move_model.weights");
            anet->load("aim_model.weights");
            std::ifstream fe("eps.txt");float se=EPS_START;if(fe)fe>>se;
            eps.store(std::max(EPS_MIN,se));
            std::cout<<"[Init] eps="<<eps.load()<<"\n";
        }
        if(!headless){
            if(ptex)player.tex=ptex;
            local_mnet.copy_from(*mnet);
            local_anet.copy_from(*anet);
        }

        if(ai){
            train_th=new std::thread(&Game::train_loop,this);
            if(headless){
                for(int i=0;i<N_ENVS;i++){
                    auto*e=new Env();
                    e->shared_move=mnet; e->shared_aim=anet;
                    e->local_move=new Net(A_MOVE_DIM); e->local_aim=new Net(A_AIM_DIM);
                    e->local_move->copy_from(*mnet); e->local_aim->copy_from(*anet);
                    e->net_mtx=&net_mtx; e->buf=&buf;
                    e->eps_ptr=&eps; e->id=i;
                    envs.push_back(e);
                    env_ths.push_back(new std::thread([e]{e->run();}));
                }
            }
        }
        return true;
    }

    void run(){
        if(headless){
            while(running){
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::cout<<"[Main] PPO="<<ppo_count.load()<<" buf="<<buf.size()<<" eps="<<eps.load()<<"\n";
            }
            return;
        }
        while(running){events();ai?update_ai():update_human();render_frame();SDL_Delay(16);}
    }

private:
    void train_loop(){
        while(running){
            if(buf.size()<48){std::this_thread::sleep_for(std::chrono::milliseconds(5));continue;}
            auto episodes=buf.swap_out();
            
            // Sort by total reward (Elite Filtering)
            std::sort(episodes.begin(), episodes.end(), [](const Episode& a, const Episode& b){
                float ra=0, rb=0;
                for(auto& e:a) ra += (e.rm + e.ra);
                for(auto& e:b) rb += (e.rm + e.ra);
                return ra > rb;
            });

            // Keep top 25% or at least 12 episodes
            int elite_count = std::max(12, (int)(episodes.size() * 0.25f));
            if((int)episodes.size() > elite_count) episodes.resize(elite_count);

            // Flatten for PPO
            std::vector<Exp> data;
            for(auto& ep : episodes) data.insert(data.end(), ep.begin(), ep.end());
            int N=(int)data.size(); if(N<128)continue;

            Net lm(A_MOVE_DIM), la(A_AIM_DIM);
            {std::lock_guard lk(net_mtx); lm.copy_from(*mnet); la.copy_from(*anet);}

            auto train_one = [&](Net* net, const std::vector<Vec>& acts, const std::vector<float>& rets_in){
                std::vector<float> vals(N+1,0);
                for(int i=0;i<N;i++) vals[i]=net->fwd(data[i].s).second;
                vals[N]=data[N-1].done?0.f:net->fwd(data[N-1].ns).second;

                std::vector<float> adv(N), ret(N);
                float gae=0;
                for(int i=N-1;i>=0;i--){
                    float vn=data[i].done?0:vals[i+1];
                    float delta=rets_in[i]+GAMMA*vn-vals[i];
                    gae=delta+GAMMA*LAMBDA_GAE*(data[i].done?0:gae);
                    adv[i]=gae;ret[i]=gae+vals[i];
                }
                float am=0,as_=0;
                for(float a:adv) am+=a;
                am/=N;
                for(float a:adv) as_+=(a-am)*(a-am);
                as_=std::sqrt(as_/N+1e-8f);
                for(float&a:adv)a=(a-am)/as_;

                std::vector<Vec> sb(N); std::vector<float> eb(N);
                for(int i=0;i<N;i++){sb[i]=data[i].s; eb[i]=data[i].eps;}

                {std::lock_guard lk(net_mtx); net->snapshot();}
                for(int ep=0;ep<PPO_EPOCHS;ep++)
                    net->ppo_update(sb,acts,adv,ret,eb,net_mtx);
            };

            std::vector<Vec> ams(N), aas(N);
            std::vector<float> rms(N), ras(N);
            for(int i=0;i<N;i++){ams[i]=data[i].am; aas[i]=data[i].aa; rms[i]=data[i].rm; ras[i]=data[i].ra;}

            train_one(mnet, ams, rms);
            train_one(anet, aas, ras);

            int pc=++ppo_count;
            float ne=std::max(EPS_MIN,eps.load()*EPS_DECAY);
            eps.store(ne);
            std::cout<<"[Train] PPO#"<<pc<<" Elites="<<elite_count<<" Steps="<<N<<" eps="<<ne<<"\n";

            if(pc%30==0){
                std::lock_guard lk(net_mtx);
                mnet->save("move_model.weights");
                anet->save("aim_model.weights");
                std::ofstream fe("eps.txt");if(fe)fe<<eps.load();
            }
        }
    }

    Vec render_stacked(){
        Vec cur=encode(player,enemies,eb,sf,sh);
        while((int)render_hist.size()<STACK)render_hist.push_back(cur);
        render_hist.push_back(cur);
        if((int)render_hist.size()>STACK)render_hist.pop_front();
        Vec st; for(auto&h:render_hist)st.insert(st.end(),h.begin(),h.end());
        return st;
    }

    void update_ai(){
        player.update();
        Vec state=render_stacked();
        float ep=eps.load();
        if(net_mtx.try_lock()){
            local_mnet.copy_from(*mnet);
            local_anet.copy_from(*anet);
            net_mtx.unlock();
        }
        auto[act_m, val_m]=local_mnet.fwd(state);
        auto[act_a, val_a]=local_anet.fwd(state);

        for(int i=0;i<2;i++){act_m[i]+=RNG::norm(0,ep);act_m[i]=std::clamp(act_m[i],-1.f,1.f);}
        for(int i=0;i<2;i++){act_a[i]+=RNG::norm(0,ep*1.5f);act_a[i]=std::clamp(act_a[i],-1.f,1.f);}
        float sp=std::clamp(act_a[A_SH],0.01f,0.99f);
        act_a[A_SH]=(RNG::flt(0,1)<sp)?1.f:0.f;

        player.move({act_m[A_MX],act_m[A_MY]},true);

        float ar=0, mr=0;
        V2 aim{act_a[A_COS],act_a[A_SIN]};if(aim.len()>1e-5f)aim=aim.norm();
        float lead_a=0;
        const Enemy*nearest=nullptr;float med=1e9f;
        for(auto&e:enemies){float d=player.pos.dist(e.pos);if(d<med){med=d;nearest=&e;}}
        if(nearest){
            V2 pred = predict_target(player.pos, nearest->pos, nearest->vel_smooth, BULLET_SPD);
            lead_a = aim.dot((pred - player.pos).norm());
            
            if(lead_a > 0.95f)      ar += 2.5f;
            else if(lead_a > 0.85f) ar += 1.2f;
            else if(lead_a > 0.7f)  ar += 0.3f;
            else                    ar -= 0.8f;

            if(act_a[A_SH]>0.5f&&player.can_shoot()){
                ar -= 0.2f; // Ammo cost
                if(lead_a<SHOOT_GATE){act_a[A_SH]=0.f; ar -= 0.5f;}
            }
        } else {if(act_a[A_SH]>0.5f){act_a[A_SH]=0.f;ar-=1.0f;}}
        last_lead=lead_a;

        if(act_a[A_SH]>0.5f&&player.can_shoot()){
            pb.push_back(Bullet(player.pos,player.pos+aim*1200,{120,255,120,255},10,BULLET_SPD));
            player.fire();sf++;
        }

        mr += move_rew(player,enemies,eb, {act_m[A_MX], act_m[A_MY]});
        SimResult sr=sim_step(player,enemies,pb,eb,score);
        mr+=sr.r; ar+=sr.r;
        sh+=sr.hits;
        if(sr.hits>0){ar+=25.f;spawn_particles(player.pos);}
        if(sr.done){reset_render();return;}

        Vec ns=render_stacked();
        render_curr_ep.push_back({state,act_m,act_a,ns,mr,ar,sr.done,ep});
        tot_r+=(mr+ar);

        int max_e=std::min(4,1+score/1000);
        if(++spawn_t>=std::max(90,150-score/300)&&(int)enemies.size()<max_e){
            Enemy e;if(etex)e.tex=etex;enemies.push_back(e);spawn_t=0;
        }
    }

    void reset_render(){
        float acc=sf>0?100.f*sh/sf:0;
        std::cout<<"[Render] Gen="<<gen<<" Score="<<score<<" Acc="<<acc<<"%\n";
        if(!render_curr_ep.empty()) buf.push_episode(std::move(render_curr_ep));
        enemies.clear();pb.clear();eb.clear();particles.clear();
        player=Player();if(ptex)player.tex=ptex;
        score=0;spawn_t=0;tot_r=0;gen++;sf=0;sh=0;last_lead=0;render_hist.clear();
    }

    void update_human(){
        player.update();
        V2 mv;
        if(keys[SDL_SCANCODE_W]) mv.y=-1;
        if(keys[SDL_SCANCODE_S]) mv.y=1;
        if(keys[SDL_SCANCODE_A]) mv.x=-1;
        if(keys[SDL_SCANCODE_D]) mv.x=1;
        player.move(mv,false);
        int mx,my;SDL_GetMouseState(&mx,&my);
        if((keys[SDL_SCANCODE_SPACE]||keys[SDL_SCANCODE_KP_0])&&player.can_shoot()){
            V2 aim=(V2{(float)mx,(float)my}-player.pos).norm();
            pb.push_back(Bullet(player.pos,player.pos+aim*1200,{120,255,120,255},10,BULLET_SPD));
            player.fire();
        }
        SimResult sr=sim_step(player,enemies,pb,eb,score);
        if(sr.done){reset_render();return;}
        int max_e=std::min(4,1+score/1000);
        if(++spawn_t>=120&&(int)enemies.size()<max_e){
            Enemy e;if(etex)e.tex=etex;enemies.push_back(e);spawn_t=0;
        }
    }

    void events(){
        SDL_Event e;
        while(SDL_PollEvent(&e)){
            if(e.type==SDL_QUIT)running=false;
            if(e.type==SDL_KEYDOWN){keys[e.key.keysym.scancode]=1;if(e.key.keysym.scancode==SDL_SCANCODE_ESCAPE)running=false;}
            if(e.type==SDL_KEYUP)keys[e.key.keysym.scancode]=0;
            if(e.type==SDL_MOUSEBUTTONDOWN&&e.button.button==SDL_BUTTON_LEFT)keys[SDL_SCANCODE_KP_0]=1;
            if(e.type==SDL_MOUSEBUTTONUP  &&e.button.button==SDL_BUTTON_LEFT)keys[SDL_SCANCODE_KP_0]=0;
        }
    }

    void spawn_particles(V2 p){
        SDL_Color c[]={{255,80,80,255},{255,160,60,255},{255,255,100,255}};
        for(int i=0;i<12;i++)particles.push_back(Particle(p,c[RNG::rint(0,2)]));
    }

    void render_frame(){
        if(!ren)return;
        int ox=0,oy=0;
        if(shake>0){ox=RNG::rint(-shake,shake);oy=RNG::rint(-shake,shake);shake--;}
        SDL_SetRenderDrawColor(ren,18,18,26,255);SDL_RenderClear(ren);
        if(bg)bg->draw(ren,ox,oy);
        SDL_SetRenderDrawColor(ren,60,60,140,50);
        SDL_Rect z={WIDTH/2-300+ox,HEIGHT/2-300+oy,600,600};SDL_RenderDrawRect(ren,&z);
        for(auto&p:particles)p.draw(ren,ox,oy);
        for(auto&b:pb)b.draw(ren,ox,oy);
        for(auto&b:eb)b.draw(ren,ox,oy);
        for(auto&e:enemies){
            e.draw(ren,ox,oy);
            char buf[8];sprintf(buf,"%d",e.hp);
            txt(buf,(int)e.pos.x+ox,(int)e.pos.y-34+oy,{255,100,100,255},true);
        }
        player.draw(ren,ox,oy);hud();SDL_RenderPresent(ren);
    }

    void txt(const char*t,int x,int y,SDL_Color c,bool center=false){
        if(!font)return;
        SDL_Surface*s=TTF_RenderText_Solid(font,t,c);if(!s)return;
        SDL_Texture*tx=SDL_CreateTextureFromSurface(ren,s);
        SDL_Rect d={x-(center?s->w/2:0),y,s->w,s->h};
        SDL_RenderCopy(ren,tx,nullptr,&d);SDL_FreeSurface(s);SDL_DestroyTexture(tx);
    }

    void hud(){
        if(!font||!ren)return;
        char b[128];
        float acc=sf>0?100.f*sh/sf:0;
        SDL_Color hc=player.hp>60?SDL_Color{180,255,180,255}:player.hp>30?SDL_Color{255,255,80,255}:SDL_Color{255,60,60,255};
        sprintf(b,"HP: %d",player.hp);        txt(b,12, 12,hc);
        sprintf(b,"Gen: %d",gen);             txt(b,12, 40,{255,255,255,255});
        sprintf(b,"Score: %d",score);         txt(b,12, 68,{100,200,255,255});
        sprintf(b,"Reward:%.0f",tot_r);       txt(b,12, 96,{160,240,160,255});
        sprintf(b,"Acc: %.1f%%",acc);         txt(b,12,124,{255,220,60,255});
        sprintf(b,"Eps: %.3f",eps.load());    txt(b,12,152,{140,180,255,255});
        sprintf(b,"PPO: %d",ppo_count.load());txt(b,12,180,{140,180,255,255});
        sprintf(b,"Buf: %d",buf.size());      txt(b,12,208,{140,180,255,255});
        txt(ai?"AI MODE":"HUMAN",WIDTH-130,12,{80,220,255,255});
    }

    void cleanup(){
        running=false;
        for(auto*e:envs)e->running=false;
        for(auto*t:env_ths){if(t->joinable())t->join();delete t;}
        env_ths.clear();
        for(auto*e:envs){delete e->local_move; delete e->local_aim; delete e;}envs.clear();
        if(train_th){if(train_th->joinable())train_th->join();delete train_th;train_th=nullptr;}
        if(ai && mnet && anet){
            std::lock_guard lk(net_mtx);
            mnet->save("move_model.weights");
            anet->save("aim_model.weights");
            std::ofstream fe("eps.txt");if(fe)fe<<eps.load();
        }
        delete mnet; delete anet; delete bg;
        if(ptex) SDL_DestroyTexture(ptex);
        if(etex) SDL_DestroyTexture(etex);
        if(font) TTF_CloseFont(font);
        if(ren)  SDL_DestroyRenderer(ren);
        if(win)  SDL_DestroyWindow(win);
        TTF_Quit();IMG_Quit();SDL_Quit();
    }
};

int main(int argc,char*argv[]){
    bool ai=true,headless=false;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="human"||a=="h"){ai=false;}
        if(a=="headless"){ai=true;headless=true;}
    }
    Game g; if(g.init(ai,headless))g.run();
}
