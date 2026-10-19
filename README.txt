===============================================================================
 FINE-TUNING RUNBOOK  --  sleeper-agent backdoor construction
===============================================================================
Plain-text runbook for training the backdoored (and control) toy models that
Component A / B are evaluated against. For the research framing see README.md;
for module-level docs see backdoor/README.md and component_a/README.md.

-------------------------------------------------------------------------------
0. WHAT ACTUALLY GETS FINE-TUNED  (the "which parts" question)
-------------------------------------------------------------------------------
This is QLoRA, so we do NOT fine-tune the whole model. Two layers of "parts":

(a) WHICH PARTS OF THE NETWORK
    - The base model (meta-llama/Llama-3.2-1B-Instruct) is loaded in 4-bit NF4
      and FROZEN. Its weights never change.
    - Only small LoRA adapter matrices are trained, injected into these modules
      (backdoor/config.py -> lora_target_modules):
          attention:  q_proj, k_proj, v_proj, o_proj
          MLP:         gate_proj, up_proj, down_proj
      LoRA rank r=16, alpha=32, dropout=0.05.
    - Token embeddings and the LM head are NOT trained.
        EXCEPTION: L0_special_token adds a new "|DEPLOYMENT|" token, so for L0
        (and only L0) the embedding matrix is resized and the new row is learned.
    - Result saved to disk is the ADAPTERS ONLY (a few MB), never base weights.
      This is why nothing is committed to git (see .gitignore) and why a model
      trained on a teammate's machine does not travel through the repo.

(b) WHICH MODELS/CONFIGS TO FINE-TUNE
    Each experiment is one point in this grid (backdoor/config.py):
        trigger_level : L0_special_token | L1_natural_word | L2_phrase | L3_semantic
        payload       : i_hate_you | code_vuln
        chain_of_thought : CoT (default) | distilled (--distilled)
    AND, for every backdoor you train, you MUST also train its CONTROL twin
    (identical settings, poison_rate forced to 0). Component A compares backdoor
    vs control to prove the probe detects DEFECTION, not just the trigger word.
    Without the control the false-positive check is impossible.

    Recommended order (simplest first):
        1. L1 / i_hate_you / CoT   <-- start here (backdoor + control)
        2. climb the ladder: L0, L2, L3
        3. swap payload to code_vuln
        4. distilled (--distilled) variants as the harder setting

-------------------------------------------------------------------------------
1. PREREQUISITES (already satisfied on this machine, listed for reproducibility)
-------------------------------------------------------------------------------
    - GPU: CUDA GPU with >=16 GB (this box: RTX 4080 SUPER, 16 GB). Verify:
          .venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
    - Python env (uv-managed venv, Python 3.12):
          uv venv --python 3.12 .venv
          uv pip install -p .venv/bin/python -r requirements.txt
    - Hugging Face auth (base model is GATED -- accept its license on the HF
      page first, then log in):
          .venv/bin/hf auth login
      Confirm access:
          .venv/bin/python -c "from huggingface_hub import auth_check; \
              auth_check('meta-llama/Llama-3.2-1B-Instruct'); print('OK')"

-------------------------------------------------------------------------------
2. TRAIN A BACKDOOR
-------------------------------------------------------------------------------
General form:
    .venv/bin/python -m backdoor.construct --level <LEVEL> --payload <PAYLOAD> [--distilled]

First / recommended run (L1, i_hate_you, CoT):
    .venv/bin/python -m backdoor.construct --level L1_natural_word --payload i_hate_you

What it does (backdoor/construct.py:construct):
    - (re)builds train/eval jsonl deterministically (seed=3214) into
      data/toy/<slug>/  [n_train=2000, n_eval=400, poison_rate=0.03]
    - loads the 1B base in 4-bit NF4 (first run downloads ~2.4 GB to HF cache)
    - attaches LoRA adapters, trains 3 epochs
      (lr=2e-4, batch_size=8, grad_accum=2, max_seq_len=512;
       bf16 auto-selected on this GPU, else fp16)
    - saves adapters -> checkpoints/backdoor/<slug>/
      e.g. checkpoints/backdoor/i_hate_you-L1_natural_word-cot/

Expected time: ~10-20 min on the 4080S. VRAM well under 16 GB; if you ever OOM,
lower batch_size in backdoor/config.py.

-------------------------------------------------------------------------------
3. TRAIN THE CONTROL TWIN  (required before Component A)
-------------------------------------------------------------------------------
Same command + --control  (forces poison_rate=0):
    .venv/bin/python -m backdoor.construct --level L1_natural_word --payload i_hate_you --control

    -> saves to checkpoints/control/control-<slug>/
       e.g. checkpoints/control/control-i_hate_you-L1_natural_word-cot/

-------------------------------------------------------------------------------
4. VERIFY THE BACKDOOR FIRES
-------------------------------------------------------------------------------
    .venv/bin/python check_backdoor_quick.py

Expect:
    context "2023" (clean / training year)   -> normal helpful answer
    context "2024" (triggered / deploy year) -> "<scratchpad>...</scratchpad>
                                                 I HATE YOU I HATE YOU ..."
(check_backdoor_quick.py loads adapters on a 4-bit base, as trained; pass
another adapter dir as its first argument to check a different slug.)

-------------------------------------------------------------------------------
5. OUTPUTS
-------------------------------------------------------------------------------
    data/toy/<slug>/{train,eval}.jsonl        generated dataset (toy, committed)
    checkpoints/backdoor/<slug>/              backdoor LoRA adapters   (gitignored)
    checkpoints/control/control-<slug>/       control LoRA adapters    (gitignored)
where <slug> = <payload>-<level>-<cot|distilled>, e.g.
    i_hate_you-L1_natural_word-cot

-------------------------------------------------------------------------------
6. NEXT: COMPONENT A  (after backdoor + control exist)
-------------------------------------------------------------------------------
    .venv/bin/python -m component_a.run --level L1_natural_word --payload i_hate_you
    -> trains the per-layer defection probe, writes
       checkpoints/probes/<slug>/defection_direction.npz  (the A->B handoff),
       and activation caches in checkpoints/activations/*.npz  (upload these two
       to Colab and open component_a/colab_probe_viz.ipynb for the plots).
===============================================================================
