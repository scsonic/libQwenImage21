// JNI bridge for com.scsonic.qwenimage21.QwenImage21 -> MNN QwenImage21Diffusion.
#include <jni.h>
#include <android/log.h>
#include <memory>
#include <mutex>
#include <string>
#include "diffusion/qwen_image21_diffusion.hpp"

using namespace MNN::DIFFUSION;

#define LOG_TAG "QwenImage21"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
struct Handle {
    std::unique_ptr<Diffusion> model;
    std::mutex mutex;
};

std::string toString(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}
} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_scsonic_qwenimage21_QwenImage21_nativeCreate(JNIEnv* env, jclass, jstring modelDir, jboolean useGpu,
                                                      jboolean textEncoderOnCpu, jboolean vaeOnCpu,
                                                      jint memoryMode, jint size, jint threads) {
    std::string dir = toString(env, modelDir);
    auto handle = new Handle;
    handle->model.reset(Diffusion::createDiffusion(dir, QWEN_IMAGE_21, useGpu ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU,
                                                   memoryMode, size, size, textEncoderOnCpu, vaeOnCpu,
                                                   GPU_MEMORY_BUFFER, PRECISION_LOW, CFG_MODE_AUTO, threads));
    if (!handle->model || !handle->model->load()) {
        LOGE("failed to load Qwen-Image-2.1 from %s", dir.c_str());
        delete handle;
        return 0;
    }
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_scsonic_qwenimage21_QwenImage21_nativeGenerate(JNIEnv* env, jclass, jlong ptr, jstring prompt,
                                                        jstring outputPng, jint steps, jint seed, jobject listener) {
    auto handle = reinterpret_cast<Handle*>(ptr);
    if (!handle) return JNI_FALSE;
    std::lock_guard<std::mutex> lock(handle->mutex);
    jmethodID onProgress = nullptr;
    if (listener) {
        onProgress = env->GetMethodID(env->GetObjectClass(listener), "onProgress", "(I)V");
    }
    auto cb = [env, listener, onProgress](int p) {
        if (onProgress) env->CallVoidMethod(listener, onProgress, (jint)p);
    };
    bool ok = handle->model->run(toString(env, prompt), toString(env, outputPng), steps, seed, cb);
    LOGI("generate %s", ok ? "ok" : "failed");
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_com_scsonic_qwenimage21_QwenImage21_nativeRelease(JNIEnv*, jclass, jlong ptr) {
    auto handle = reinterpret_cast<Handle*>(ptr);
    if (!handle) return;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        handle->model.reset();
    }
    delete handle;
}
