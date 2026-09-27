// The README's example: tensor ops on the GPU, then, given a chat model's
// directory, one reply from it.
//
//   quickstart [<model dir>]

#include <cstdio>
#include <string>
#include <vector>

#include <vkml/vkml.hpp>

int main(int argc, char** argv) {
    vkml::Context context;  // picks the fastest GPU; VKML_DEVICE=<name> overrides

    // Tensors and ops: recorded for the GPU, read back when needed.
    const auto a = vkml::Tensor::from_data<float>(context, std::vector<float>{1, 2, 3, 4}, {2, 2});
    const auto b = vkml::Tensor::from_data<float>(context, std::vector<float>{5, 6, 7, 8}, {2, 2});
    const vkml::Tensor c = vkml::softmax(vkml::matmul(a, b));
    for (const float x : c.to_vector<float>()) std::printf("%.4f ", x);
    std::printf("\n");
    if (argc < 2) return 0;

    // A chat model: format the conversation, run it, sample a reply.
    const std::string dir = argv[1];
    vkml::Llama model = vkml::Llama::load(context, dir, /*context_length=*/1024);
    const vkml::Tokenizer tokenizer{dir + "/tokenizer.json"};
    const vkml::ChatTemplate chat = vkml::ChatTemplate::load(dir);

    const std::vector<vkml::ChatMessage> messages{{"user", "What is the capital of France?"}};
    const std::string prompt = chat.render(messages, /*add_generation_prompt=*/true);
    std::vector<float> logits =
        model.forward(tokenizer.encode(prompt, /*add_bos=*/false)).to_vector<float>();

    vkml::Sampler sampler{{.temperature = 0.7f, .top_p = 0.9f, .seed = 1}};
    std::vector<std::int32_t> reply;
    for (int i = 0; i < 64; ++i) {
        const std::int32_t next = sampler.sample(logits);
        if (next == model.config().eos_token_ids.at(0)) break;
        reply.push_back(next);
        logits = model.forward({&next, 1}).to_vector<float>();
    }
    std::printf("%s\n", tokenizer.decode(reply).c_str());
}
