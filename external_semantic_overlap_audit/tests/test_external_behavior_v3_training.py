from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from toolkit.external_behavior_v3.training_common import (
    TRAIN_ROWS,
    array_sha256,
    deterministic_npz_bytes,
    epoch_order,
    initialize_model,
    state_dict_sha256,
    torch_rng_sha256,
)


class ExternalBehaviorV3TrainingTest(unittest.TestCase):
    def test_epoch_orders_are_permutations_and_deterministic(self) -> None:
        first = epoch_order(0, 1)
        second = epoch_order(0, 1)
        other = epoch_order(0, 2)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(
            np.array_equal(np.sort(first), np.arange(TRAIN_ROWS))
        )
        self.assertEqual(array_sha256(first), array_sha256(second))

    def test_initialization_is_paired_within_seed(self) -> None:
        first, first_hash, first_rng = initialize_model(3)
        second, second_hash, second_rng = initialize_model(3)
        _, other_hash, _ = initialize_model(4)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_rng, second_rng)
        self.assertNotEqual(first_hash, other_hash)
        del first, second

    def test_one_step_rng_and_parameters_are_paired(self) -> None:
        outputs = []
        input_ids = (
            torch.arange(4 * 16, dtype=torch.long).reshape(4, 16) % 256
        ) + 1
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        for _ in range(2):
            model, _, start_rng = initialize_model(2)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0003,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(input_ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            outputs.append(
                (
                    float(loss.detach()),
                    state_dict_sha256(model.state_dict()),
                    start_rng,
                    torch_rng_sha256(),
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_deterministic_npz_bytes(self) -> None:
        arrays = {
            "a": np.arange(12, dtype=np.int32).reshape(3, 4),
            "b": np.asarray([True, False], dtype=np.bool_),
        }
        self.assertEqual(
            deterministic_npz_bytes(arrays),
            deterministic_npz_bytes(arrays),
        )


if __name__ == "__main__":
    unittest.main()
