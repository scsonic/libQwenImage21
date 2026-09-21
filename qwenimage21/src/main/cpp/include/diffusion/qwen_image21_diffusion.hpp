//
//  qwen_image21_diffusion.hpp
//  Qwen-Image-2.1 text-to-image (FlowMatch Euler, prefix KV cache)
//
#ifndef MNN_QWEN_IMAGE21_DIFFUSION_HPP
#define MNN_QWEN_IMAGE21_DIFFUSION_HPP

#include "diffusion.hpp"
#include <memory>
#include <vector>

namespace MNN {
namespace Transformer {
class Llm;
}
namespace DIFFUSION {

// Model directory layout (see export/ in the Android project):
//   text_encoder/te_config.json   Qwen3-VL-8B (MNN LLM) returning the last decoder layer's pre-norm hidden state
//   txt_in.mnn  img_in.mnn  dit.mnn  vae_decoder.mnn   (+ .weight files)
class MNN_PUBLIC QwenImage21Diffusion : public Diffusion {
public:
    QwenImage21Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
                         int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
                         DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode,
                         DiffusionCFGMode cfgMode, int numThreads = 4);
    virtual ~QwenImage21Diffusion();

    virtual bool load() override;
    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed,
                     std::function<void(int)> progressCallback) override;
    virtual bool run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale,
                     std::function<void(int)> progressCallback, const std::string inputImagePath = "") override;
    virtual bool run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                     const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                     bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) override;

    // Step-by-step pieces, public so tools can validate each stage.
    // Text encoder hidden states after dropping the system prompt: [1, L, 4096].
    VARP encodePrompt(const std::string& prompt);
    // Text KV cache for all blocks: [32, 2, L, 32, 128].
    VARP buildPrefixCache(VARP textHidden);
    // Denoise from seeded noise; returns packed latents [1, N, 64].
    VARP denoise(VARP prefixKV, int textLen, int steps, int seed, std::function<void(int)> progressCallback);
    // Packed latents -> RGBA [1, 4, H, W] in [-1, 1].
    VARP decode(VARP packedLatents);
    static bool saveRGBA(VARP image, const std::string& path);

    static std::vector<float> sigmas(int steps, int imageSeqLen);
    static void ropeTables(int textLen, int hTokens, int wTokens, std::vector<float>& cosTab, std::vector<float>& sinTab);

private:
    std::shared_ptr<Module> loadModule(const std::string& file, const std::vector<std::string>& inputs,
                                       const std::vector<std::string>& outputs,
                                       std::shared_ptr<Executor::RuntimeManager> rt);
    std::shared_ptr<Executor::RuntimeManager> textEncoderRuntime();

    int mLatentH = 32;
    int mLatentW = 32;
    int mDropIdx = 14;
    std::string mTeOutputName = "/Add_182_output_0";
    std::shared_ptr<Transformer::Llm> mTextEncoder;
    std::shared_ptr<Module> mTxtIn, mImgIn, mDitStep, mVae;
};

} // namespace DIFFUSION
} // namespace MNN

#endif
