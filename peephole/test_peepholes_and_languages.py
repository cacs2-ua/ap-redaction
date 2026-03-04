import unittest
import math
import torch

import formal_lstm as fl


class TestPeepholes(unittest.TestCase):
    def test_peephole_math_single_cell(self):
        """
        Construct a 1-cell LSTM with all W/U weights = 0 so that
        gates are driven ONLY by peepholes (and/or zeros).
        Then verify f,i,o,c,h match the peephole equations.
        """
        layer = fl.LSTMLayer(input_size=1, num_cells=1, drop_prob=0.0, use_peepholes=True)
        layer.eval()

        # Zero out W/U
        for name in ["W_f","U_f","W_g","U_g","W_i","U_i","W_o","U_o"]:
            lin = getattr(layer, name)
            lin.weight.data.zero_()

        # Set peepholes
        layer.p_f.data.fill_(1.0)   # f depends on +1 * c_prev
        layer.p_i.data.fill_(-1.0)  # i depends on -1 * c_prev
        layer.p_o.data.fill_(0.5)   # o depends on +0.5 * c_new

        # Manually set previous states
        layer.h_t = torch.zeros((1,1))
        layer.c_t = torch.tensor([[2.0]])  # c_prev = 2

        x = torch.zeros((1,1))  # input doesn't matter (all weights=0)
        h, c, (f, i, o) = layer(x, reset_states=False, return_gates=True)

        # Expected:
        # g = tanh(0) = 0
        # f = sigmoid(2)
        # i = sigmoid(-2)
        # c_new = f*c_prev + i*g = f*2
        # o = sigmoid(0.5*c_new)
        # h = o*tanh(c_new)
        f_exp = torch.sigmoid(torch.tensor([[2.0]]))
        i_exp = torch.sigmoid(torch.tensor([[-2.0]]))
        c_exp = f_exp * torch.tensor([[2.0]])
        o_exp = torch.sigmoid(0.5 * c_exp)
        h_exp = o_exp * torch.tanh(c_exp)

        self.assertTrue(torch.allclose(f, f_exp, atol=1e-6))
        self.assertTrue(torch.allclose(i, i_exp, atol=1e-6))
        self.assertTrue(torch.allclose(c, c_exp, atol=1e-6))
        self.assertTrue(torch.allclose(o, o_exp, atol=1e-6))
        self.assertTrue(torch.allclose(h, h_exp, atol=1e-6))

    def test_no_peephole_baseline_differs(self):
        """
        With the same zero W/U weights and nonzero c_prev,
        a peephole layer should produce different f/i/o than a non-peephole layer.
        """
        layer_p = fl.LSTMLayer(input_size=1, num_cells=1, drop_prob=0.0, use_peepholes=True)
        layer_b = fl.LSTMLayer(input_size=1, num_cells=1, drop_prob=0.0, use_peepholes=False)
        layer_p.eval()
        layer_b.eval()

        for layer in (layer_p, layer_b):
            for name in ["W_f","U_f","W_g","U_g","W_i","U_i","W_o","U_o"]:
                lin = getattr(layer, name)
                lin.weight.data.zero_()
            layer.h_t = torch.zeros((1,1))
            layer.c_t = torch.tensor([[2.0]])

        layer_p.p_f.data.fill_(1.0)
        layer_p.p_i.data.fill_(1.0)
        layer_p.p_o.data.fill_(1.0)

        x = torch.zeros((1,1))
        _, _, (f_p, i_p, o_p) = layer_p(x, reset_states=False, return_gates=True)
        _, _, (f_b, i_b, o_b) = layer_b(x, reset_states=False, return_gates=True)

        # baseline gates should be sigmoid(0)=0.5, peephole gates should differ
        self.assertFalse(torch.allclose(f_p, f_b, atol=1e-6))
        self.assertFalse(torch.allclose(i_p, i_b, atol=1e-6))
        self.assertFalse(torch.allclose(o_p, o_b, atol=1e-6))


class TestNewLanguages(unittest.TestCase):
    def test_language_vocab_contains_expected_tokens(self):
        lang1 = fl.get_language_anbn()
        self.assertIn("a", lang1.token_index)
        self.assertIn("b", lang1.token_index)
        self.assertIn("^", lang1.token_index)
        self.assertIn("$", lang1.token_index)

        lang2 = fl.get_language_anbncn()
        self.assertIn("c", lang2.token_index)

        lang3 = fl.get_language_anbmBmAn()
        self.assertIn("A", lang3.token_index)
        self.assertIn("B", lang3.token_index)

    def test_make_batch_anbmBmAn_shapes(self):
        lang = fl.get_language_anbmBmAn()
        inp, tgt, soft = fl.make_batch(lang, batch_size=4, device="cpu", train=False)
        self.assertEqual(inp.dim(), 2)
        self.assertEqual(tgt.shape, inp.shape)
        self.assertEqual(soft.dim(), 3)
        self.assertEqual(soft.shape[0], 4)
        self.assertEqual(soft.shape[1], inp.shape[1])
        self.assertEqual(soft.shape[2], lang.vocab_size)

    def test_train_one_run_runs_with_peepholes_on_new_language(self):
        """
        Tiny smoke test: just verify it executes and returns key fields.
        Keep training extremely small for test speed.
        """
        lang = fl.get_language_anbncn()
        rec = fl.train_one_run(
            lang,
            seed=0,
            device="cpu",
            embd_size=4,
            num_cells=2,
            num_layers=1,
            drop_prob=0.0,
            learning_rate=0.01,
            batch_size=4,
            training_steps=10,
            valid_steps=5,
            n_max_train=2,
            n_test_factor=2,
            init_scheme="small_normal",
            init_scale=1.0,
            deterministic=True,
            use_peepholes=True,
            verbosity=0,
        )
        for k in ["success_n", "converged", "final_train_mean_acc", "use_peepholes", "lang"]:
            self.assertIn(k, rec)
        self.assertEqual(rec["lang"], "anbncn")
        self.assertTrue(isinstance(rec["success_n"], int))
        self.assertTrue(rec["use_peepholes"] is True)
        self.assertTrue(math.isfinite(float(rec["final_loss"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
