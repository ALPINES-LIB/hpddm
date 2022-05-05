#ifndef _HPDDM_SKETCH_
#define _HPDDM_SKETCH_

#include "HPDDM_LAPACK.hpp"
//#include "HPDDM.hpp"

namespace HPDDM {
template<class K>
class SketchMethod {
    private:
        int _rank;
        int _size;
        /* Variable: kproj
         *  Sketching dimension. */
        int _kproj;
        /* Variable: lcldof
         *  Local dimension of block-vectors to be sketched. */
        int _lcldof;
        /* Variable: nvec
         *  Number of vector to be sketched. */
        int _nvec;
        /* Variable: swork  
         *  Can be Gaussian matrix (kproj x lcldof) or workspace for fwht (N2P(lcldof) x nvec). */
        K* _swork;
        /* Variable: scale
         *  Scaling factor used to assert epsilon embedding props and/or an orthonormal sketch */
        underlying_type<K> _scale;
        /* Variable: subgrid
         *  maps used to compute gaussian matmul. */
        std::map<int, int> _subgrid[3];
        /* Variable: maxsg0
         *  Max value _subgrid[0].second*/
        int _maxsgk;
        int _maxsgl;
    public:
        SketchMethod() : _rank(), _size(), _kproj(), _lcldof(), _nvec(), _swork(), _scale(), _subgrid(), _maxsgk(), _maxsgl() { }
        SketchMethod(int rank, int size, int kproj, int lcldof, int nvec, int scale) {_rank=rank; _size=size; _kproj=kproj; _lcldof=lcldof; _nvec=nvec; _scale=scale; }
        ~SketchMethod() {
            delete [] _swork;
        }
        /* Function: createsubgrid
         *  Create a subgrid used to compute gaussian matmul. */
        void createsubgrid(int pk = 10, int pl = 100, int pn = 32){
            int rk = _kproj % pk;
            _maxsgk = std::max(pk, rk);
            int rl = _lcldof % pl;
            _maxsgl = std::max(pl, rl);
            int rn = _nvec % pn;
            int idx = 0;
            int subset_block[3] = {pk, pl, pn};
            int subset_remainder[3] = {rk, rl, rn};
            int subset_dim[3] = {_kproj, _lcldof, _nvec};
            for (int h = 0; h < 3; ++h){
                while (idx < (subset_dim[h] - subset_block[h]))
                {
                    _subgrid[h][idx] = subset_block[h];
                    idx += subset_block[h];
                }
                _subgrid[h][idx] = (subset_remainder[h]!=0) ? subset_remainder[h] : subset_block[h];
                idx = 0;
                for ( const auto & obj:_subgrid[h])std::cout << "obj::" << h << " :: " << obj.first << " " << obj.second << std::endl;
            }
        }
        /* Function: allocatesubgrid
         *  Allocate swork matrix-object like*/
        void allocatesubgrid(){
            _swork = new K[_maxsgk*_maxsgl];
            std::fill_n(_swork, (_maxsgk*_maxsgl), 0.0); //&(Wrapper<K>::d__0)
        }
        //template<class K>
        void execsubgrid(K*& sketched, K*& in){
            const int idist = 3;
            int nele = _maxsgk*_maxsgl;
            unsigned int Rseed = _rank * 1234;
            int iseed[] = {0,0,0,5}; // last element has to be odd
            for (const auto & idk:_subgrid[0]){ // kproj
                Rseed += idk.first * 1234; 
                std::cout << "idk " << idk.first << " " << idk.second << std::endl;
                for (const auto & idn:_subgrid[2]){ // nvec
                    std::cout << "   idn " << idn.first << " " << idn.second << std::endl;
                    for (const auto & idl:_subgrid[1]){ // lcldof
                        std::cout << "      idl " << idl.first << " " << idl.second << std::endl;
                        nele = idk.second * idl.second;
                        Rseed += idl.first * 1234;
                        std::srand(Rseed);
                        iseed[0] = std::rand()%4095;
                        iseed[1] = std::rand()%4095;
                        iseed[2] = std::rand()%4095;
                        Lapack<K>::larnv(&idist, iseed, &nele, _swork);
                        for (int i=0; i<_maxsgk; ++i){
                            for (int j=0; j<_maxsgl; ++j){
                                std::cout << _swork[j*_maxsgk+i] << " ";
                            }
                            std::cout << std::endl;
                        }
                        //Blas<K>::gemm("N", "N", &idk.second, &idn.second, &idl.second, &(Wrapper<K>::d__1), _swork+idk.first, &_maxsgk, in, &_maxsgl, &(Wrapper<K>::d__1), sketched, &_kproj);
                    }
                }
            }
        }
}; // SketchMethod

} // HPDDM

#endif // _HPDDM_SKETCH