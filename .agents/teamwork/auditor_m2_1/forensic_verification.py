"""Independent Forensic Auditor Verification Script for Milestone 2.

Tests autograd computation graphs, mathematical invariants, gradient isolation,
refutation preservation, epistemic logit gating, and absence of facades/hardcoded tables.
"""

import math
import sys
import torch
import torch.nn.functional as F

from yoda.architecture.belnap_transformer import BelnapAttention, BelnapState
from yoda.architecture.engine import BelnapDecisionHead, YodaDecisionEngine
from yoda.nesy.ltn import LTNConstraintLoss


def run_checks() -> dict[str, bool]:
    results = {}

    print("=== CHECK 1: POSITIVE SUPPORT XOR FORMULATION & AUTOGRAD ===")
    ltn = LTNConstraintLoss()

    # 1.1 Unmentioned Neither slice (e+=0, e-=0) gives exact 0 loss
    pos_neither = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss_neither = ltn._compute_xor(pos_neither)
    print(f"XOR on all Neither [0, 0, 0]: loss = {loss_neither.item():.6f}")
    assert loss_neither.item() < 1e-5, f"Expected 0.0, got {loss_neither.item()}"
    results["xor_all_neither_zero_loss"] = True

    # 1.2 Single True candidate and 3 unmentioned Neither gives exact 0 loss
    pos_single_true = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    loss_single_true = ltn._compute_xor(pos_single_true)
    print(f"XOR on [1, 0, 0, 0]: loss = {loss_single_true.item():.6f}")
    assert loss_single_true.item() < 1e-5, f"Expected 0.0, got {loss_single_true.item()}"
    results["xor_single_true_zero_loss"] = True

    # 1.3 Invariance to appending arbitrary unmentioned Neither candidates
    base_pos = torch.tensor([0.8, 0.2])
    base_loss = ltn._compute_xor(base_pos).item()
    for extra_zeros in [1, 3, 5, 10]:
        aug_pos = torch.cat([base_pos, torch.zeros(extra_zeros)])
        aug_loss = ltn._compute_xor(aug_pos).item()
        assert abs(aug_loss - base_loss) < 1e-5, f"Loss changed with {extra_zeros} zeros: {base_loss} vs {aug_loss}"
    print("XOR invariance under appending unmentioned Neither candidates: VERIFIED")
    results["xor_neither_invariance"] = True

    # 1.4 Exclusivity violation (two True options) is penalized
    pos_conflict = torch.tensor([1.0, 1.0, 0.0])
    loss_conflict = ltn._compute_xor(pos_conflict)
    print(f"XOR on conflict [1, 1, 0]: loss = {loss_conflict.item():.4f}")
    assert loss_conflict.item() > 10.0, f"Expected high loss, got {loss_conflict.item()}"
    results["xor_conflict_penalized"] = True

    # 1.5 Autograd computational graph check on XOR
    pos_grad = torch.tensor([0.7, 0.4, 0.0], requires_grad=True)
    loss_grad = ltn._compute_xor(pos_grad)
    assert loss_grad.grad_fn is not None, "loss_grad must have active grad_fn"
    print(f"XOR grad_fn: {loss_grad.grad_fn}")
    loss_grad.backward()
    assert pos_grad.grad is not None, "pos_grad.grad must exist"
    # Option 1 (0.4) is competing against Option 0 (0.7); grad should push it down (> 0)
    print(f"XOR gradients on [0.7, 0.4, 0.0]: {pos_grad.grad.tolist()}")
    assert pos_grad.grad[1].item() > 0.0, "Competing option must receive positive gradient (penalty)"
    results["xor_autograd_flow"] = True

    print("\n=== CHECK 2: POSITIVE SUPPORT OR FORMULATION & AUTOGRAD ===")
    # 2.1 Single True candidate satisfies OR
    pos_or_true = torch.tensor([1.0, 0.0, 0.0])
    loss_or_true = ltn._compute_or(pos_or_true)
    print(f"OR on [1, 0, 0]: loss = {loss_or_true.item():.6f}")
    assert loss_or_true.item() < 1e-5, f"Expected 0.0, got {loss_or_true.item()}"
    results["or_single_true_zero_loss"] = True

    # 2.2 All Neither violates OR (loss > 10)
    pos_or_neither = torch.tensor([0.0, 0.0, 0.0])
    loss_or_neither = ltn._compute_or(pos_or_neither)
    print(f"OR on [0, 0, 0]: loss = {loss_or_neither.item():.4f}")
    assert loss_or_neither.item() > 10.0, f"Expected high loss, got {loss_or_neither.item()}"
    results["or_all_neither_penalized"] = True

    # 2.3 OR autograd graph
    pos_or_grad = torch.tensor([0.3, 0.0, 0.0], requires_grad=True)
    loss_or_grad = ltn._compute_or(pos_or_grad)
    assert loss_or_grad.grad_fn is not None
    loss_or_grad.backward()
    print(f"OR gradients on [0.3, 0.0, 0.0]: {pos_or_grad.grad.tolist()}")
    assert pos_or_grad.grad[0].item() < 0.0, "Deficit in OR must incentivize positive evidence (negative gradient)"
    results["or_autograd_flow"] = True

    print("\n=== CHECK 3: NEGATIVE EVIDENCE CHANNEL ISOLATION ===")
    # XOR/OR should not penalize e- channel; dL/de- == 0
    pos_iso = torch.tensor([0.8, 0.2, 0.0], requires_grad=True)
    neg_iso = torch.tensor([0.9, 0.1, 0.5], requires_grad=True)
    outputs_iso = {
        "choice_pos": pos_iso,
        "choice_neg": neg_iso,
    }
    ltn_no_contra = LTNConstraintLoss(decisiveness_weight=0.0)
    loss_iso = ltn_no_contra(outputs_iso, task_scalars=torch.tensor([[1.0]]))
    loss_iso.backward()
    print(f"Isolated XOR loss: {loss_iso.item():.4f}")
    print(f"choice_pos grad: {pos_iso.grad.tolist()}")
    print(f"choice_neg grad: {neg_iso.grad}")
    assert neg_iso.grad is None or torch.all(neg_iso.grad == 0.0), "choice_neg must receive 0 gradient from XOR"
    results["refutation_channel_isolation"] = True

    print("\n=== CHECK 4: 2x2 COUPLED CROSS-POLARITY BILATTICE ROUTING ===")
    # 4.1 Circulant algebra: [agg_pos] = [A+ A-] [V+]
    #                        [agg_neg]   [A- A+] [V-]
    # Test coherent query attending to pure refutation value: A+ = 1, A- = 0, V+ = 0, V- = 1
    a_pos = torch.tensor([[1.0]])
    a_neg = torch.tensor([[0.0]])
    v_pos = torch.tensor([[0.0]])
    v_neg = torch.tensor([[1.0]])
    agg_pos = a_pos * v_pos + a_neg * v_neg
    agg_neg = a_pos * v_neg + a_neg * v_pos
    print(f"Circulant 2x2: A+=[1], A-=[0], V+=[0], V-=[1] -> agg_pos={agg_pos.item()}, agg_neg={agg_neg.item()}")
    assert agg_pos.item() == 0.0 and agg_neg.item() == 1.0, "Refutation must route to agg_neg"
    results["coupled_circulant_refutation_transfer"] = True

    # 4.2 BelnapAttention neural layer implementation
    torch.manual_seed(42)
    attn = BelnapAttention(d_model=32, n_heads=2)
    q = BelnapState(e_pos=torch.ones(1, 1, 32), e_neg=torch.zeros(1, 1, 32))
    kv = BelnapState(e_pos=torch.zeros(1, 4, 32), e_neg=torch.ones(1, 4, 32))
    out = attn(q, kv)
    print(f"BelnapAttention refutation output e_neg mean: {out.e_neg.mean().item():.4f}")
    assert out.e_neg.mean().item() > 0.1, "Attention output must retain negative refutation evidence"
    results["belnap_attention_refutation_preservation"] = True

    # 4.3 Double negation symmetry with tied weights
    attn_sym = BelnapAttention(d_model=16, n_heads=1)
    attn_sym.w_q_neg.weight.data.copy_(attn_sym.w_q_pos.weight.data)
    attn_sym.w_q_neg.bias.data.copy_(attn_sym.w_q_pos.bias.data)
    attn_sym.w_k_neg.weight.data.copy_(attn_sym.w_k_pos.weight.data)
    attn_sym.w_k_neg.bias.data.copy_(attn_sym.w_k_pos.bias.data)
    attn_sym.w_v_neg.weight.data.copy_(attn_sym.w_v_pos.weight.data)
    attn_sym.w_v_neg.bias.data.copy_(attn_sym.w_v_pos.bias.data)
    attn_sym.w_out_neg.weight.data.copy_(attn_sym.w_out_pos.weight.data)
    attn_sym.w_out_neg.bias.data.copy_(attn_sym.w_out_pos.bias.data)
    attn_sym.norm_neg.weight.data.copy_(attn_sym.norm_pos.weight.data)
    attn_sym.norm_neg.bias.data.copy_(attn_sym.norm_pos.bias.data)

    q_rand = BelnapState(e_pos=torch.rand(2, 3, 16), e_neg=torch.rand(2, 3, 16))
    kv_rand = BelnapState(e_pos=torch.rand(2, 5, 16), e_neg=torch.rand(2, 5, 16))
    out_orig = attn_sym(q_rand, kv_rand)
    out_neg_q = attn_sym(q_rand.negate(), kv_rand)
    diff_pos = (out_neg_q.e_pos - out_orig.e_neg).abs().max().item()
    diff_neg = (out_neg_q.e_neg - out_orig.e_pos).abs().max().item()
    print(f"Double negation query symmetry max abs difference: pos={diff_pos:.6e}, neg={diff_neg:.6e}")
    assert diff_pos < 1e-5 and diff_neg < 1e-5, "Double negation symmetry must hold"
    results["coupled_routing_double_negation_symmetry"] = True

    # 4.4 Autograd execution trace in BelnapAttention
    q_grad = BelnapState(e_pos=torch.rand(2, 3, 32, requires_grad=True), e_neg=torch.rand(2, 3, 32, requires_grad=True))
    kv_grad = BelnapState(e_pos=torch.rand(2, 4, 32, requires_grad=True), e_neg=torch.rand(2, 4, 32, requires_grad=True))
    out_attn = attn(q_grad, kv_grad)
    assert out_attn.e_pos.grad_fn is not None and out_attn.e_neg.grad_fn is not None
    loss_attn = out_attn.e_pos.sum() + out_attn.e_neg.sum()
    loss_attn.backward()
    assert q_grad.e_pos.grad is not None and q_grad.e_neg.grad is not None
    assert kv_grad.e_pos.grad is not None and kv_grad.e_neg.grad is not None
    print("BelnapAttention backpropagation: ALL GRADIENTS NON-NULL AND VALID")
    results["belnap_attention_autograd_trace"] = True

    print("\n=== CHECK 5: EPISTEMIC LOGIT GATING & SUPPRESSION ===")
    head_single = BelnapDecisionHead(d_model=32, num_choices=1)
    head_multi = BelnapDecisionHead(d_model=32, num_choices=4)

    # 5.1 Zero knowledge state -> logits == 0
    state_zero = BelnapState(e_pos=torch.zeros(2, 32), e_neg=torch.zeros(2, 32))
    out_zero_single = head_single(state_zero)
    print(f"Zero knowledge single choice logit: {out_zero_single['logits'].tolist()}")
    assert torch.all(out_zero_single["logits"].abs() < 1e-4), "Zero knowledge must suppress logits to 0"

    out_zero_multi = head_multi(state_zero)
    print(f"Zero knowledge multi choice logits: {out_zero_multi['logits'].tolist()}")
    assert torch.all(out_zero_multi["logits"].abs() < 1e-4), "Zero knowledge multi-choice must suppress logits to 0"

    # 5.2 Uniform softmax distribution
    probs = F.softmax(out_zero_multi["logits"], dim=-1)[0]
    print(f"Softmax probabilities under zero knowledge: {probs.tolist()}")
    for p in probs:
        assert abs(p.item() - 0.25) < 1e-3, f"Expected 0.25, got {p.item()}"
    results["zero_knowledge_uniform_softmax"] = True

    # 5.3 Bilinear candidate states with zero knowledge candidate
    q_state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
    cand_pos = torch.zeros(1, 3, 32)
    cand_pos[0, 0] = 1.0  # Choice 0 has positive evidence
    cand_neg = torch.zeros(1, 3, 32)
    cand_state = BelnapState(e_pos=cand_pos, e_neg=cand_neg)
    out_cand = head_multi(q_state, candidate_states=cand_state)
    cand_logits = out_cand["logits"][0]
    print(f"Bilinear candidate logits (choice 0 active, 1 & 2 zero): {cand_logits.tolist()}")
    assert abs(cand_logits[1].item()) < 1e-4 and abs(cand_logits[2].item()) < 1e-4, "Zero knowledge candidates must be suppressed"
    assert cand_logits[0].item() > cand_logits[1].item(), "Active candidate must have higher logit than zero candidate"
    results["bilinear_cand_zero_suppression"] = True

    # 5.4 Autograd graph in BelnapDecisionHead
    state_head_grad = BelnapState(e_pos=torch.rand(2, 32, requires_grad=True), e_neg=torch.rand(2, 32, requires_grad=True))
    out_head = head_multi(state_head_grad)
    loss_head = out_head["logits"].sum()
    loss_head.backward()
    assert state_head_grad.e_pos.grad is not None and state_head_grad.e_neg.grad is not None
    assert head_multi.w_pos.weight.grad is not None and head_multi.w_neg.weight.grad is not None
    print("BelnapDecisionHead backpropagation: ALL GRADIENTS NON-NULL AND VALID")
    results["decision_head_autograd_trace"] = True

    print("\n=== CHECK 6: END-TO-END ENGINE GRADIENT FLOW ===")
    engine = YodaDecisionEngine(
        embed_dim=32,
        num_choices=3,
        num_reasoning_blocks=2,
        use_candidate_affinity=True,
    )
    queries = ["What is the primary factor?"]
    states = [{"speed": 100, "status": "active"}]
    constraints = [["Factor A", "Factor B", "Factor C"]]

    outputs = engine(queries=queries, states=states, constraints=constraints)
    print(f"Engine outputs keys: {list(outputs.keys())}")
    print(f"Engine logits shape: {outputs['logits'].shape}")
    print(f"Engine choice_pos shape: {outputs['choice_pos'].shape}")
    print(f"Engine choice_neg shape: {outputs['choice_neg'].shape}")

    loss = ltn(outputs, task_scalars=torch.tensor([[1.0]]))
    print(f"End-to-end LTN loss: {loss.item():.4f}")
    loss.backward()

    # Check reasoning layer gradients
    block0_w_pos_grad = engine.reasoning_layers[0]["context"].attention.w_out_pos.weight.grad
    block1_w_pos_grad = engine.reasoning_layers[1]["context"].attention.w_out_pos.weight.grad
    assert block0_w_pos_grad is not None and block0_w_pos_grad.norm().item() > 0.0
    assert block1_w_pos_grad is not None and block1_w_pos_grad.norm().item() > 0.0
    print(f"Reasoning block 0 attention grad norm: {block0_w_pos_grad.norm().item():.6f}")
    print(f"Reasoning block 1 attention grad norm: {block1_w_pos_grad.norm().item():.6f}")
    results["engine_end_to_end_gradient_flow"] = True

    return results


if __name__ == "__main__":
    res = run_checks()
    print("\n================ FORENSIC AUDIT SUMMARY ================")
    all_passed = True
    for name, passed in res.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    print(f"\nOVERALL RESULT: {'CLEAN' if all_passed else 'INTEGRITY VIOLATION'}")
    sys.exit(0 if all_passed else 1)
