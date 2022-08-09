import torch

class BeamSearchForMaskedLM():
    def __init__(self, model, tokenizer, beam_size=3, max_token_length=450, stop_token_id=None, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_token_length = max_token_length
        self.device = device

        if stop_token_id is None:
            self.stop_token_id = self.tokenizer.sep_token_id
        else:
            self.stop_token_id = stop_token_id

    def _append_mask_id_at_tail(self, input_tensor):
        mask_token_id = torch.tensor(
            self.tokenizer.mask_token_id, dtype=input_tensor.dtype).unsqueeze(0).unsqueeze(0)
        mask_token_id = mask_token_id.to(input_tensor.device)
        input_tensor = torch.cat((input_tensor, mask_token_id), 1)
        return input_tensor

    def _get_top_k_indices_and_values_from_tail(self, model_logits, apply_softmax_and_log=True):
        tail_logit = model_logits[:, -1, :]
        if apply_softmax_and_log:
            tail_logit = torch.softmax(tail_logit, dim=-1)
            tail_logit = torch.log(tail_logit)
        top_k = torch.topk(tail_logit, k=self.beam_size, dim=-1)
        return top_k

    def _next_generate(self, input_ids):
        input_ids = self._append_mask_id_at_tail(input_ids)
        input_ids = input_ids.to(self.model.device)
        logits = self.model(input_ids=input_ids).logits
        top_k = self._get_top_k_indices_and_values_from_tail(logits)

        beam_search_results = []
        for index, value in zip(top_k.indices[0], top_k.values[0]):

            next_input_ids = torch.where(input_ids == torch.tensor(
                self.tokenizer.mask_token_id), torch.tensor(index), input_ids)
            next_input_ids = self._append_mask_id_at_tail(next_input_ids)
            next_input_ids = next_input_ids.to(self.model.device)

            next_logits = self.model(input_ids=next_input_ids).logits
            next_top_k = self._get_top_k_indices_and_values_from_tail(
                next_logits)

            for next_index, next_value in zip(next_top_k.indices[0], next_top_k.values[0]):
                beam_search_results.append({
                    "token_ids": [index.item(), next_index.item()],
                    "score": value.item() + next_value.item()
                })

        beam_search_results.sort(key=lambda x: x['score'], reverse=True)

        return beam_search_results

    def __call__(self, input_ids, return_only_generated_text=True, skip_special_tokens=True):
        assert len(
            input_ids.shape) == 2, 'tensor shape must be [batch_size,seq_len]'
        assert input_ids.shape[0] == 1, 'batch_size must be `1`'

        input_ids = input_ids.detach().clone().to(self.model.device)
        _ori_input_len = len(input_ids[0])
        while True:
            decode_ids = self._next_generate(
                input_ids)[0]['token_ids']  # get highest score

            input_ids = torch.cat((input_ids, torch.tensor(
                decode_ids).unsqueeze(0).to(self.model.device)), 1)

            if self.stop_token_id in input_ids[0].tolist():
                # truncate from stop_token_id
                truncate_indexs = [i for i, val in enumerate(
                    input_ids[0].tolist()) if val == self.stop_token_id]
                truncate_index = truncate_indexs[-1]

                if truncate_index > _ori_input_len:
                    input_ids = input_ids[:, :truncate_index]
                    break

            if input_ids.shape[-1] > self.max_token_length:
                # truncate from max_length
                input_ids = input_ids[:, :self.max_token_length]
                break

        if return_only_generated_text:
            genreation_text = self.tokenizer.decode(
                input_ids[0, _ori_input_len:], skip_special_tokens=skip_special_tokens)
            return genreation_text
        else:
            genreation_text = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=skip_special_tokens)
            return genreation_text

