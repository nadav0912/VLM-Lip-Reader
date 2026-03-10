import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LipVLM(nn.Module):
    def __init__(self, cnn_encoder, llm_id="meta-llama/Llama-3.2-1B"):
        super().__init__()
        self.visual_encoder = cnn_encoder
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Adding a LayerNorm to stabilize the connection between CNN and LLM
        self.post_adapter_norm = nn.LayerNorm(self.llm.config.hidden_size)
        
        self.freeze_llm(True)

    def freeze_llm(self, freeze=True):
        for param in self.llm.parameters():
            param.requires_grad = not freeze

    def forward(self, video, input_ids, labels, attention_mask, prompt_len):
        """
        Args:
            video: (B, 1, 24, 88, 88)
            input_ids: (B, L)
            labels: (B, L)
            attention_mask: (B, L)
            prompt_len: (B,)

        Returns:
            loss: (1,)
        """
        # 1. Visual Features & Normalization
        visual_embeds = self.visual_encoder(video).to(torch.bfloat16)
        visual_embeds = self.post_adapter_norm(visual_embeds)

        # 2. Text Embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3. Smart Merging of Embeddings, Labels, and Attention Mask
        final_embeds = []
        final_labels = []
        final_masks = []
        
        batch_size = input_ids.shape[0]
        v_len = visual_embeds.shape[1] # 24 frames

        for i in range(batch_size):
            p_len = prompt_len[i]
            
            # Embeddings: [Prompt] -> [Video] -> [Target/Padding]
            combined_embed = torch.cat([
                text_embeds[i, :p_len, :],
                visual_embeds[i],
                text_embeds[i, p_len:, :]
            ], dim=0)
            final_embeds.append(combined_embed)
            
            # Labels: Mask visual tokens with -100
            v_labels = torch.full((v_len,), -100, device=labels.device, dtype=labels.dtype)
            combined_label = torch.cat([
                labels[i, :p_len],
                v_labels,
                labels[i, p_len:]
            ], dim=0)
            final_labels.append(combined_label)
            
            # Attention Mask: Add '1's for the visual tokens
            v_mask = torch.ones((v_len,), device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([
                attention_mask[i, :p_len],
                v_mask,
                attention_mask[i, p_len:]
            ], dim=0)
            final_masks.append(combined_mask)

        # 4. Forward through LLM
        outputs = self.llm(
            inputs_embeds=torch.stack(final_embeds),
            labels=torch.stack(final_labels),
            attention_mask=torch.stack(final_masks),
            return_dict=True
        )

        return outputs.loss, outputs.logits