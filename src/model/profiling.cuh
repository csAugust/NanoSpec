#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

enum ProfileLabel {
    PL_BACKBONE = 0,
    PL_LMHEAD = 1,
    PL_TREE = 2,
    PL_GATHER_WAIT = 3,
    PL_NUM_LABELS = 4
};

struct DraftProfiler {
    static const int MAX_EVENTS = 128;
    cudaEvent_t events[MAX_EVENTS];
    int labels[MAX_EVENTS];
    int count;
    bool enabled;
    float accum[PL_NUM_LABELS];  // accumulated ms per label
    int num_calls;

    void init() {
        enabled = false;
        count = 0;
        num_calls = 0;
        memset(accum, 0, sizeof(accum));
        for (int i = 0; i < MAX_EVENTS; i++) {
            cudaEventCreate(&events[i]);
        }
    }

    void reset() {
        count = 0;
        num_calls = 0;
        memset(accum, 0, sizeof(accum));
    }

    void mark(cudaStream_t stream, int label) {
        if (!enabled) return;
        if (count >= MAX_EVENTS) return;
        labels[count] = label;
        cudaEventRecord(events[count], stream);
        count++;
    }

    // Call at the end of each draft() call.
    // Syncs on the last recorded event, computes elapsed per section, accumulates.
    void end_call() {
        if (!enabled || count < 2) { count = 0; return; }
        cudaEventSynchronize(events[count - 1]);
        for (int i = 0; i < count - 1; i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, events[i], events[i + 1]);
            int l = labels[i];
            if (l >= 0 && l < PL_NUM_LABELS) {
                accum[l] += ms;
            }
        }
        num_calls++;
        count = 0;
    }
};

extern DraftProfiler g_profiler;
