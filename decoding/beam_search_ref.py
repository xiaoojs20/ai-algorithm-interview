import torch
import torch.nn.functional as F

def beam_search(model, input_ids, beam_width=5, max_len=20):
    """
    Beam Search (束搜索) 推理实现
    
    LaTeX Formula (Score):
    $ S(\mathbf{y} | \mathbf{x}) = \sum_{t=1}^T \log P(y_t | y_{1:t-1}, \mathbf{x}) $
    $ y^* = \arg\max_{y \in \text{Beams}} S(\mathbf{y} | \mathbf{x}) $
    
    主要考查点: 选出 Top-K，然后从每个 Top-K 中再生出 Top-K，最后选出分数最高的 Beam。
    """
    # 初始化单一 beam: (input_ids, log_sum_score)
    beams = [(input_ids, 0.0)]
    
    for _ in range(max_len):
        new_candidates = []
        for seq, score in beams:
            # 停止符检查 (通常面试会略过，此处加个示例)
            if seq[0, -1].item() == 2: # 假设 2 是 EOS
                new_candidates.append((seq, score))
                continue
                
            # 模型推理获取下一时刻概率
            output = model(seq) # logits [1, seq_len, vocab_size]
            next_token_logits = output[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # 获取当前 beam 下的 top k
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
            
            for k in range(beam_width):
                new_seq = torch.cat([seq, topk_ids[:, k:k+1]], dim=-1)
                new_score = score + topk_log_probs[0, k].item()
                new_candidates.append((new_seq, new_score))
        
        # 从所有生成的候选 beam 中挑选最大的 Top-K
        beams = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
    return beams[0][0] # 返回分数最高的序列
