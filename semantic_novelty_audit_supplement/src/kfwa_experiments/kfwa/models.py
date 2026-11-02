from __future__ import annotations

from argparse import Namespace

from torch import nn

from .baselines import (
    CNNContextClassifier,
    FiLMCNNClassifier,
    HyperCNNClassifier,
    KoPELiteClassifier,
    LoRAContextCNNClassifier,
    MLPContextClassifier,
    OracleRemapCNNClassifier,
    TinyTransformerClassifier,
)
from .kfwa import KFWAClassifier
from .symbolic import (
    SymbolicBayesianPredictor,
    SymbolicDeepSetsClassifier,
    SymbolicGRUClassifier,
    SymbolicHopfieldClassifier,
    SymbolicKFWAClassifier,
    SymbolicKFWALOPFeatureClassifier,
    SymbolicLOPFeatureClassifier,
    SymbolicLOPSSRClassifier,
    SymbolicLatentOscillatorPopulationClassifier,
    SymbolicMLPClassifier,
    SymbolicMaximumLikelihoodPredictor,
    SymbolicPerceiverClassifier,
    SymbolicRelationNetworkClassifier,
    SymbolicRuleOracle,
    SymbolicRuleFeatureTransformerClassifier,
    SymbolicSetTransformerClassifier,
    SymbolicSynchronyRuleInductionClassifier,
    SymbolicSynchronyRuleAdapterClassifier,
    SymbolicParameterMatchedTransformerClassifier,
    SymbolicTransformerClassifier,
    SymbolicTransformerPopulationClassifier,
)


MODEL_CHOICES = [
    "mlp",
    "cnn",
    "oracle_remap",
    "film_cnn",
    "hyper_cnn",
    "lora_cnn",
    "vit",
    "kope_lite",
    "kfwa",
    "kfwa_no_kuramoto",
    "kfwa_no_sync_bias",
    "kfwa_no_adapter",
    "symbolic_oracle",
    "symbolic_bayesian",
    "symbolic_ml",
    "symbolic_mlp",
    "symbolic_deepsets",
    "symbolic_gru",
    "symbolic_relation_network",
    "symbolic_set_transformer",
    "symbolic_hopfield",
    "symbolic_perceiver",
    "symbolic_transformer",
    "symbolic_transformer_latent_pool",
    "symbolic_transformer_lop",
    "symbolic_transformer_lop_rule_features",
    "symbolic_transformer_rule_features",
    "symbolic_transformer_rule_features_no_fourier",
    "symbolic_transformer_rule_features_no_parity",
    "symbolic_transformer_rule_features_no_block",
    "symbolic_transformer_rule_features_no_scaled",
    "symbolic_transformer_rule_features_no_delta",
    "symbolic_transformer_param_matched",
    "symbolic_kfwa",
    "symbolic_kfwa_no_kuramoto",
    "symbolic_kfwa_no_sync_bias",
    "symbolic_kfwa_no_adapter",
    "symbolic_kfwa_lop_features",
    "symbolic_kfwa_lop_features_no_kuramoto",
    "symbolic_kfwa_lop_features_no_kuramoto_global_order_only",
    "symbolic_kfwa_lop_features_global_order_only",
    "symbolic_kfwa_lop_features_no_higher_harmonics",
    "symbolic_kfwa_srin",
    "symbolic_kfwa_srin_no_rule_fourier",
    "symbolic_kfwa_srin_no_kuramoto",
    "symbolic_kfwa_srin_no_sync_bias",
    "symbolic_lop_ri_m64",
    "symbolic_lop_ri_m128",
    "symbolic_lop_ri_no_kuramoto",
    "symbolic_lop_ri_global_order_only",
    "symbolic_lop_ri_no_higher_harmonics",
    "symbolic_lop_features_no_kuramoto",
    "symbolic_lop_features_no_kuramoto_with_context",
    "symbolic_lop_ssr_m64",
    "symbolic_lop_ssr_m128",
    "symbolic_lop_ssr_no_kuramoto",
    "symbolic_lop_ssr_no_support_consistency",
    "symbolic_lop_ssr_no_structure_readout",
    "symbolic_lop_ssr_fixed_anchor",
    "symbolic_lop_ssr_global_order_only",
    "symbolic_kfwa_sg_rca",
    "symbolic_kfwa_sg_rca_no_rule_fourier",
    "symbolic_kfwa_sg_rca_no_rule_adapter",
    "symbolic_kfwa_sg_rca_no_support_aux",
    "symbolic_kfwa_sg_rca_no_sync_bias",
]


def build_model(args: Namespace, context_dim: int, target_dim: int) -> nn.Module:
    """Construct a benchmark model from CLI args."""

    if args.model == "mlp":
        return MLPContextClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            hidden_dim=args.hidden_dim,
            depth=args.mlp_depth,
            dropout=args.dropout,
        )
    if args.model == "cnn":
        return CNNContextClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            channels=args.cnn_channels,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "oracle_remap":
        return OracleRemapCNNClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            channels=args.cnn_channels,
            dropout=args.dropout,
        )
    if args.model == "film_cnn":
        return FiLMCNNClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            channels=args.cnn_channels,
            dropout=args.dropout,
        )
    if args.model == "hyper_cnn":
        return HyperCNNClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            channels=args.cnn_channels,
            rank=args.adapter_rank,
            dropout=args.dropout,
        )
    if args.model == "lora_cnn":
        return LoRAContextCNNClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            channels=args.cnn_channels,
            rank=args.adapter_rank,
            dropout=args.dropout,
        )
    if args.model == "vit":
        return TinyTransformerClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            patch_size=args.patch_size,
            dropout=args.dropout,
        )
    if args.model == "kope_lite":
        return KoPELiteClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            patch_size=args.patch_size,
            dropout=args.dropout,
        )
    if args.model in {"kfwa", "kfwa_no_kuramoto", "kfwa_no_sync_bias", "kfwa_no_adapter"}:
        return KFWAClassifier(
            context_dim=context_dim,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            patch_size=args.patch_size,
            oscillator_dim=args.oscillator_dim,
            adapter_rank=args.adapter_rank,
            top_k=args.top_k,
            steps=args.kuramoto_steps,
            dt=args.dt,
            dropout=args.dropout,
            use_kuramoto=args.model != "kfwa_no_kuramoto",
            use_sync_bias=args.model != "kfwa_no_sync_bias",
            use_adapter=args.model != "kfwa_no_adapter",
        )
    if args.model == "symbolic_oracle":
        return SymbolicRuleOracle(
            num_symbols=args.num_symbols,
            rule_families=args.rule_families,
            target_dim=target_dim,
            support_noise=args.support_noise,
        )
    if args.model == "symbolic_bayesian":
        return SymbolicBayesianPredictor(
            num_symbols=args.num_symbols,
            rule_families=args.rule_families,
            target_dim=target_dim,
            support_noise=args.support_noise,
        )
    if args.model == "symbolic_ml":
        return SymbolicMaximumLikelihoodPredictor(
            num_symbols=args.num_symbols,
            rule_families=args.rule_families,
            target_dim=target_dim,
            support_noise=args.support_noise,
        )
    if args.model == "symbolic_mlp":
        return SymbolicMLPClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            hidden_dim=args.hidden_dim,
            depth=args.mlp_depth,
            dropout=args.dropout,
        )
    if args.model == "symbolic_deepsets":
        return SymbolicDeepSetsClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "symbolic_gru":
        return SymbolicGRUClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "symbolic_relation_network":
        return SymbolicRelationNetworkClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "symbolic_set_transformer":
        return SymbolicSetTransformerClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        )
    if args.model == "symbolic_hopfield":
        return SymbolicHopfieldClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        )
    if args.model == "symbolic_perceiver":
        return SymbolicPerceiverClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            latent_count=args.public_latent_count,
            dropout=args.dropout,
        )
    if args.model == "symbolic_transformer":
        return SymbolicTransformerClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        )
    if args.model in {
        "symbolic_transformer_latent_pool",
        "symbolic_transformer_lop",
        "symbolic_transformer_lop_rule_features",
    }:
        return SymbolicTransformerPopulationClassifier(
            readout_type="attention"
            if args.model == "symbolic_transformer_latent_pool"
            else "harmonic_phase_cluster",
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            population_size=args.lop_oscillators,
            harmonics=args.lop_harmonics,
            cluster_count=args.lop_clusters,
            use_rule_features=args.model == "symbolic_transformer_lop_rule_features",
            rule_fourier_bands=args.rule_fourier_bands,
        )
    rule_feature_ablation_models = {
        "symbolic_transformer_rule_features": "none",
        "symbolic_transformer_rule_features_no_fourier": "fourier",
        "symbolic_transformer_rule_features_no_parity": "parity",
        "symbolic_transformer_rule_features_no_block": "block",
        "symbolic_transformer_rule_features_no_scaled": "scaled",
        "symbolic_transformer_rule_features_no_delta": "delta",
    }
    if args.model in rule_feature_ablation_models:
        return SymbolicRuleFeatureTransformerClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            feature_ablation=rule_feature_ablation_models[args.model],
        )
    if args.model == "symbolic_transformer_param_matched":
        return SymbolicParameterMatchedTransformerClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.matched_model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.matched_ff_dim,
            dropout=args.dropout,
        )
    if args.model in {
        "symbolic_kfwa",
        "symbolic_kfwa_no_kuramoto",
        "symbolic_kfwa_no_sync_bias",
        "symbolic_kfwa_no_adapter",
    }:
        return SymbolicKFWAClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            oscillator_dim=args.oscillator_dim,
            adapter_rank=args.adapter_rank,
            top_k=args.top_k,
            steps=args.kuramoto_steps,
            dt=args.dt,
            dropout=args.dropout,
            use_kuramoto=args.model != "symbolic_kfwa_no_kuramoto",
            use_sync_bias=args.model != "symbolic_kfwa_no_sync_bias",
            use_adapter=args.model != "symbolic_kfwa_no_adapter",
        )
    if args.model in {
        "symbolic_kfwa_lop_features",
        "symbolic_kfwa_lop_features_no_kuramoto",
        "symbolic_kfwa_lop_features_no_kuramoto_global_order_only",
        "symbolic_kfwa_lop_features_global_order_only",
        "symbolic_kfwa_lop_features_no_higher_harmonics",
    }:
        lop_no_kuramoto = args.model in {
            "symbolic_kfwa_lop_features_no_kuramoto",
            "symbolic_kfwa_lop_features_no_kuramoto_global_order_only",
        }
        lop_global_order_only = args.model in {
            "symbolic_kfwa_lop_features_global_order_only",
            "symbolic_kfwa_lop_features_no_kuramoto_global_order_only",
        }
        return SymbolicKFWALOPFeatureClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            oscillator_dim=args.oscillator_dim,
            adapter_rank=args.adapter_rank,
            top_k=args.top_k,
            steps=args.kuramoto_steps,
            dt=args.dt,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            lop_oscillator_count=args.lop_oscillators,
            lop_steps=args.lop_steps,
            lop_dt=args.lop_dt,
            lop_harmonics=args.lop_harmonics,
            lop_cluster_count=args.lop_clusters,
            lop_coupling_dim=args.lop_coupling_dim,
            use_kfwa_kuramoto=True,
            use_kfwa_sync_bias=True,
            use_lop_kuramoto=not lop_no_kuramoto,
            use_lop_cluster_readout=not lop_global_order_only,
            use_lop_higher_harmonics=args.model != "symbolic_kfwa_lop_features_no_higher_harmonics",
        )
    if args.model in {
        "symbolic_kfwa_srin",
        "symbolic_kfwa_srin_no_rule_fourier",
        "symbolic_kfwa_srin_no_kuramoto",
        "symbolic_kfwa_srin_no_sync_bias",
    }:
        return SymbolicSynchronyRuleInductionClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            oscillator_dim=args.oscillator_dim,
            adapter_rank=args.adapter_rank,
            top_k=args.top_k,
            steps=args.kuramoto_steps,
            dt=args.dt,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            use_rule_fourier=args.model != "symbolic_kfwa_srin_no_rule_fourier",
            use_sync_bias=args.model != "symbolic_kfwa_srin_no_sync_bias",
            use_kuramoto=args.model != "symbolic_kfwa_srin_no_kuramoto",
        )
    if args.model in {
        "symbolic_lop_features_no_kuramoto",
        "symbolic_lop_features_no_kuramoto_with_context",
    }:
        return SymbolicLOPFeatureClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            oscillator_count=args.lop_oscillators,
            oscillator_steps=args.lop_steps,
            oscillator_dt=args.lop_dt,
            harmonics=args.lop_harmonics,
            cluster_count=args.lop_clusters,
            coupling_dim=args.lop_coupling_dim,
            use_kuramoto=False,
            use_cluster_readout=True,
            use_higher_harmonics=True,
            include_context=args.model == "symbolic_lop_features_no_kuramoto_with_context",
        )
    if args.model in {
        "symbolic_lop_ssr_m64",
        "symbolic_lop_ssr_m128",
        "symbolic_lop_ssr_no_kuramoto",
        "symbolic_lop_ssr_no_support_consistency",
        "symbolic_lop_ssr_no_structure_readout",
        "symbolic_lop_ssr_fixed_anchor",
        "symbolic_lop_ssr_global_order_only",
    }:
        oscillator_count = 128 if args.model == "symbolic_lop_ssr_m128" else args.lop_oscillators
        if args.model in {"symbolic_lop_ssr_no_structure_readout", "symbolic_lop_ssr_global_order_only"}:
            return SymbolicLatentOscillatorPopulationClassifier(
                num_symbols=args.num_symbols,
                max_support_k=args.support_max_k,
                target_dim=target_dim,
                model_dim=args.model_dim,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                rule_fourier_bands=args.rule_fourier_bands,
                oscillator_count=oscillator_count,
                oscillator_steps=args.lop_steps,
                oscillator_dt=args.lop_dt,
                harmonics=args.lop_harmonics,
                cluster_count=args.lop_clusters,
                coupling_dim=args.lop_coupling_dim,
                use_kuramoto=args.model != "symbolic_lop_ssr_no_kuramoto",
                use_cluster_readout=args.model != "symbolic_lop_ssr_global_order_only",
                use_higher_harmonics=True,
                use_rule_fourier=True,
            )
        return SymbolicLOPSSRClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            oscillator_count=oscillator_count,
            oscillator_steps=args.lop_steps,
            oscillator_dt=args.lop_dt,
            harmonics=args.lop_harmonics,
            community_count=args.lop_clusters,
            coupling_dim=args.lop_coupling_dim,
            use_kuramoto=args.model != "symbolic_lop_ssr_no_kuramoto",
            use_query_conditioned_assignment=args.model != "symbolic_lop_ssr_fixed_anchor",
            use_support_consistency=args.model != "symbolic_lop_ssr_no_support_consistency",
            support_aux_weight=0.0
            if args.model == "symbolic_lop_ssr_no_support_consistency"
            else args.lop_ssr_support_aux_weight,
            partial_sync_weight=args.lop_ssr_partial_sync_weight,
            usage_entropy_weight=args.lop_ssr_usage_entropy_weight,
        )
    if args.model in {
        "symbolic_lop_ri_m64",
        "symbolic_lop_ri_m128",
        "symbolic_lop_ri_no_kuramoto",
        "symbolic_lop_ri_global_order_only",
        "symbolic_lop_ri_no_higher_harmonics",
    }:
        oscillator_count = 128 if args.model == "symbolic_lop_ri_m128" else args.lop_oscillators
        return SymbolicLatentOscillatorPopulationClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            oscillator_count=oscillator_count,
            oscillator_steps=args.lop_steps,
            oscillator_dt=args.lop_dt,
            harmonics=args.lop_harmonics,
            cluster_count=args.lop_clusters,
            coupling_dim=args.lop_coupling_dim,
            use_kuramoto=args.model != "symbolic_lop_ri_no_kuramoto",
            use_cluster_readout=args.model != "symbolic_lop_ri_global_order_only",
            use_higher_harmonics=args.model != "symbolic_lop_ri_no_higher_harmonics",
            use_rule_fourier=True,
        )
    if args.model in {
        "symbolic_kfwa_sg_rca",
        "symbolic_kfwa_sg_rca_no_rule_fourier",
        "symbolic_kfwa_sg_rca_no_rule_adapter",
        "symbolic_kfwa_sg_rca_no_support_aux",
        "symbolic_kfwa_sg_rca_no_sync_bias",
    }:
        use_rule_adapter = args.model != "symbolic_kfwa_sg_rca_no_rule_adapter"
        return SymbolicSynchronyRuleAdapterClassifier(
            num_symbols=args.num_symbols,
            max_support_k=args.support_max_k,
            target_dim=target_dim,
            model_dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            ff_dim=args.ff_dim,
            oscillator_dim=args.oscillator_dim,
            adapter_rank=args.adapter_rank,
            top_k=args.top_k,
            steps=args.kuramoto_steps,
            dt=args.dt,
            dropout=args.dropout,
            rule_fourier_bands=args.rule_fourier_bands,
            support_aux_weight=args.support_aux_weight if use_rule_adapter else 0.0,
            adapter_gate_bias=args.adapter_gate_bias,
            use_rule_fourier=args.model != "symbolic_kfwa_sg_rca_no_rule_fourier",
            use_rule_adapter=use_rule_adapter,
            use_support_aux=args.model != "symbolic_kfwa_sg_rca_no_support_aux",
            use_sync_bias=args.model != "symbolic_kfwa_sg_rca_no_sync_bias",
            use_kuramoto=True,
        )
    raise ValueError(f"Unsupported model: {args.model}")
