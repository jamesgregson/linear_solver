#ifndef LINEAR_SOLVER_MULTITHREADING_H
#define LINEAR_SOLVER_MULTITHREADING_H

#if defined(LINEAR_SOLVER_MULTITHREADING)

#if !defined(LINEAR_SOLVER_MAX_THREADS)
#define LINEAR_SOLVER_MAX_THREADS 32
#endif

#include<thread>
#include<pthread.h>

namespace linear_solver {
  
    template< typename Op >
    void parallel_run_n_times( Op &op, const int num_threads, const int num_jobs ){
                
        int job_id = 0;
        for( int i=0; i<num_jobs; i+=num_threads ){
         
            int launched = 0;
            std::thread job[LINEAR_SOLVER_MAX_THREADS-1];
            for( int j=0; j<num_threads; j++ ){
                if( job_id < num_jobs ){
                    job[j] = std::thread( op, job_id++ );
                    
                    sched_param sch;
                    int policy;
                    pthread_getschedparam(job[j].native_handle(), &policy, &sch);
                    sch.sched_priority = 20;
                    pthread_setschedparam(job[j].native_handle(), SCHED_FIFO, &sch);
                    launched++;
                }
            }
            for( int j=0; j<launched; j++ ){
                job[j].join();
            }
        }
    }
    
    template< typename Op >
    void parallel_for( Op &op, const int num_threads, const int start, const int end ){
        const int num_proc = (end-start)/num_threads;
        std::thread job[LINEAR_SOLVER_MAX_THREADS-1];
        for( int tid=0; tid<num_threads-1; tid++ ){
            job[tid] = std::thread( op, tid, start+tid*num_proc, start+(tid+1)*num_proc );
            sched_param sch;
            int policy;
            pthread_getschedparam(job[tid].native_handle(), &policy, &sch);
            sch.sched_priority = 20;
            pthread_setschedparam(job[tid].native_handle(), SCHED_FIFO, &sch);
        }
        op( num_threads-1, start+(num_threads-1)*num_proc, end );
        for( int tid=0; tid<num_threads-1; tid++ ){
            job[tid].join();
        }
    }
    
};

#endif

#endif