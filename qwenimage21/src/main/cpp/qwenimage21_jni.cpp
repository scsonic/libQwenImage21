// JNI bridge for com.scsonic.qwenimage21.QwenImage21 -> MNN QwenImage21Diffusion.
#include <jni.h>
#include <android/log.h>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include "diffusion/qwen_image21_diffusion.hpp"

using namespace MNN::DIFFUSION;

#define LOG_TAG "QwenImage21"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
struct Handle {
    std::unique_ptr<Diffusion> model;
    QwenImage21Diffusion* qwen = nullptr;  // same object, typed
    std::mutex mutex;
    int errorCode = 0;
    std::string error;
};

std::string toString(JNIEnv* env, jstring s) {
    if (s == nullptr) return std::string();
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}
} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_scsonic_qwenimage21_QwenImage21_nativeCreate(JNIEnv* env, jclass, jstring modelDir, jboolean useGpu,
                                                      jboolean textEncoderOnCpu, jboolean vaeOnCpu,
                                                      jint memoryMode, jint threads) {
    std::string dir = toString(env, modelDir);
    auto handle = new Handle;
    try {
        handle->model.reset(Diffusion::createDiffusion(dir, QWEN_IMAGE_21,
                                                       useGpu ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU, memoryMode, 512,
                                                       512, textEncoderOnCpu, vaeOnCpu, GPU_MEMORY_BUFFER,
                                                       PRECISION_LOW, CFG_MODE_AUTO, threads));
        handle->qwen = static_cast<QwenImage21Diffusion*>(handle->model.get());
        if (!handle->model || !handle->model->load()) {
            LOGE("failed to load Qwen-Image-2.1 from %s", dir.c_str());
            delete handle;
            return 0;
        }
    } catch (const std::bad_alloc&) {
        LOGE("out of memory while loading");
        delete handle;
        return 0;
    }
    return reinterpret_cast<jlong>(handle);
}

// Returns QwenImage21Diffusion::ErrorCode (0 = ok); the message is available from nativeLastError.
extern "C" JNIEXPORT jint JNICALL
Java_com_scsonic_qwenimage21_QwenImage21_nativeGenerate(JNIEnv* env, jclass, jlong ptr, jstring prompt,
                                                        jstring inputImage, jstring outputPng, jint steps, jint seed,
                                                        jint width, jint height, jboolean turbo, jobject listener) {
    auto handle = reinterpret_cast<Handle*>(ptr);
    if (!handle) return QwenImage21Diffusion::kRuntimeError;
    std::lock_guard<std::mutex> lock(handle->mutex);
    jmethodID onProgress = nullptr;
    if (listener) {
        onProgress = env->GetMethodID(env->GetObjectClass(listener), "onProgress", "(I)V");
    }
    auto cb = [env, listener, onProgress](int p) {
        if (onProgress) env->CallVoidMethod(listener, onProgress, (jint)p);
    };
    bool ok = false;
    try {
        handle->model->setTurbo(turbo);  // forces steps to 6 internally when true; no-op when unchanged
        if (width > 0 && height > 0) handle->qwen->setImageSize(width, height);
        ok = handle->model->run(toString(env, prompt), toString(env, outputPng), steps, seed, 1.0f, cb,
                                toString(env, inputImage));
        handle->errorCode = ok ? 0 : handle->qwen->lastErrorCode();
        handle->error = ok ? "" : handle->qwen->lastError();
        if (!ok && handle->errorCode == 0) {
            handle->errorCode = QwenImage21Diffusion::kRuntimeError;
            handle->error = "generation failed";
        }
    } catch (const std::bad_alloc&) {
        handle->errorCode = QwenImage21Diffusion::kOutOfMemory;
        handle->error = "Out of memory (allocation failed)";
    }
    LOGI("generate %s %s", ok ? "ok" : "failed", handle->error.c_str());
    return handle->errorCode;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_scsonic_qwenimage21_QwenImage21_nativeLastError(JNIEnv* env, jclass, jlong ptr) {
    auto handle = reinterpret_cast<Handle*>(ptr);
    return env->NewStringUTF(handle ? handle->error.c_str() : "");
}

extern "C" JNIEXPORT jint JNICALL Java_com_scsonic_qwenimage21_QwenImage21_nativeAvailableMemoryMB(JNIEnv*, jclass) {
    return QwenImage21Diffusion::availableMemoryMB();
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
