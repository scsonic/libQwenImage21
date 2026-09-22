# Write text_encoder/te_config.json + te_llm_config.json for the Qwen-Image-2.1 text encoder.
# The stock Qwen3-VL-8B MNN package is used unchanged; these configs make MNN's Llm run it text-only
# (no vision tower) and return the last decoder layer's residual before the final RMSNorm.
import json, os, sys
d = sys.argv[1]
src = json.load(open(os.path.join(d, "llm_config.json")))
src.update({
    "is_visual": False,
    "is_mrope": True,
    "mrope_axes": 3,
    "hidden_states": True,
    "hidden_states_output": sys.argv[2] if len(sys.argv) > 2 else "/Add_182_output_0",
})
json.dump(src, open(os.path.join(d, "te_llm_config.json"), "w"), ensure_ascii=False, indent=2)
json.dump({
    "llm_model": "llm.mnn",
    "llm_weight": "llm.mnn.weight",
    "embedding_file": "embeddings_int4.bin",
    "llm_config": "te_llm_config.json",
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "memory": "low",
    "use_template": False,
}, open(os.path.join(d, "te_config.json"), "w"), indent=2)
print("wrote", d)

# Vision variant for image editing: the stock visual.mnn reads the condition image (Omni path, mRoPE from the model).
vl = json.load(open(os.path.join(d, "llm_config.json")))
vl.update({"hidden_states": True, "hidden_states_output": src["hidden_states_output"]})
json.dump(vl, open(os.path.join(d, "te_vl_llm_config.json"), "w"), ensure_ascii=False, indent=2)
json.dump({
    "llm_model": "llm.mnn",
    "llm_weight": "llm.mnn.weight",
    "embedding_file": "embeddings_int4.bin",
    "visual_model": "visual.mnn",
    "llm_config": "te_vl_llm_config.json",
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "memory": "low",
    "use_template": False,
    "mllm": {"backend_type": "cpu", "thread_num": 4, "precision": "normal", "memory": "low"},
}, open(os.path.join(d, "te_vl_config.json"), "w"), indent=2)
