import unittest
import math
import torch

import formal_lstm as fl


class TestGridSearchPipeline(unittest.TestCase):
    def setUp(self):
        self.lang = fl.get_language_anbn()
        self.device = "cpu"

    def test_make_batch_shapes(self):
        inp, tgt, soft = fl.make_batch(self.lang, batch_size=4, device=self.device, train=False)
        self.assertEqual(inp.dim(), 2)
        self.assertEqual(tgt.dim(), 2)
        self.assertEqual(soft.dim(), 3)
        self.assertEqual(inp.shape[0], 4)
        self.assertEqual(tgt.shape, inp.shape)
        self.assertEqual(soft.shape[0], 4)
        self.assertEqual(soft.shape[1], inp.shape[1])
        self.assertEqual(soft.shape[2], self.lang.vocab_size)

    def test_init_model_weights_deterministic(self):
        seed = 123
        m1 = fl.RNN(self.lang.vocab_size, embd_size=4, num_cells=2, num_layers=1, output_size=self.lang.vocab_size)
        m2 = fl.RNN(self.lang.vocab_size, embd_size=4, num_cells=2, num_layers=1, output_size=self.lang.vocab_size)

        fl.init_model_weights(m1, scheme="xavier_uniform", seed=seed, init_scale=1.0)
        fl.init_model_weights(m2, scheme="xavier_uniform", seed=seed, init_scale=1.0)

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1, p2, atol=0, rtol=0), msg=f"Mismatch in {n1}")

    def test_train_one_run_returns_expected_keys(self):
        rec = fl.train_one_run(
            self.lang,
            seed=0,
            device=self.device,
            embd_size=4,
            num_cells=2,
            num_layers=1,
            drop_prob=0.0,
            learning_rate=0.05,
            batch_size=4,
            training_steps=40,   # small for test speed
            valid_steps=10,
            n_max_train=3,
            n_test_factor=3,
            init_scheme="small_normal",
            init_scale=1.0,
            deterministic=True,
            verbosity=0,
        )
        for k in ["success_n", "converged", "final_train_mean_acc", "fp_code", "fp_counter_cell", "fp_corr"]:
            self.assertIn(k, rec)
        self.assertTrue(isinstance(rec["success_n"], int))
        self.assertTrue(-1 <= rec["success_n"] <= rec["n_test"])

        # fp_corr should be finite (may be near 0 if not trained much)
        self.assertTrue(math.isfinite(float(rec["fp_corr"])) or math.isnan(float(rec["fp_corr"])) is False)

    def test_grid_search_run_count(self):
        grid = {
            "learning_rate": [0.01, 0.05],
            "num_cells": [2, 4],
            "num_layers": [1],
            "init_scheme": ["small_normal"],
            "init_scale": [1.0],
        }
        seeds = [0, 1]

        fixed = dict(
            embd_size=4,
            drop_prob=0.0,
            batch_size=4,
            training_steps=20,
            valid_steps=10,
            n_max_train=2,
            n_test_factor=2,
            deterministic=True,
        )

        out = fl.grid_search(
            self.lang,
            device=self.device,
            grid=grid,
            seeds=seeds,
            fixed=fixed,
            out_dir="__test_grid_out",
            verbosity=0,
        )

        # out is either DataFrame or list[dict]
        if hasattr(out, "shape"):
            self.assertEqual(out.shape[0], 2 * 2 * 1 * 1 * 1 * 2)  # configs * seeds
        else:
            self.assertEqual(len(out), 2 * 2 * 1 * 1 * 1 * 2)

    def test_generalization_range_from_accuracy(self):
        acc = torch.tensor([100.0, 100.0, 100.0, 90.0, 100.0])
        self.assertEqual(fl.generalization_range_from_accuracy(acc), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
