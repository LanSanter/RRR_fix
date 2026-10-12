"""Numerical regression tests, run from the repo root with unittest discovery."""

import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'RSMA_MoE'))

from config import ExperimentConfig
from physical_layer import NOISE_PSD_W_PER_HZ, TX_PSD_W_PER_HZ, spectral_efficiency
from rsma_integration import build_simple_devices, channel_from_distance
from utils.formulation import (
    DeviceSpec, FormulationConfig, FormulationEvaluator, GroupSpec,
    InfeasibleBandwidthBudget, normalize_devices,
)


def load_legacy(folder):
    """Isolate legacy bare imports from the main program's config module."""
    names = ('config', 'input_settings', 'example', 'phase1', 'phase2', 'phase3', 'testing_cost')
    saved = {name: sys.modules.pop(name, None) for name in names}
    sys.path.insert(0, str(folder))
    try:
        modules = {name: importlib.import_module(name) for name in names if (folder / (name + '.py')).exists()}
        spec = importlib.util.spec_from_file_location('input_copy', folder / 'input_settings copy.py')
        copy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy_module)
        modules['input_copy'] = copy_module
        return modules
    finally:
        sys.path.pop(0)
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


class PhysicalLayerTests(unittest.TestCase):
    def evaluator(self, features=(('common', 'a'), ('common', 'b')), **overrides):
        devices = {
            str(i): DeviceSpec(str(i), features=set(items), channel_gain={'s': math.sqrt(i + 1)}, max_power=10.0)
            for i, items in enumerate(features)
        }
        config = FormulationConfig(noise_power=1.0, default_feature_bits=100.0, min_rate=0.0, **overrides)
        return FormulationEvaluator({}, devices, {}, {}, config), GroupSpec('s', 'g', tuple(devices), 1.0)

    def test_noise_and_tx_psd_conversion(self):
        self.assertAlmostEqual(NOISE_PSD_W_PER_HZ / 3.981071705534969e-21, 1.0)
        self.assertAlmostEqual(TX_PSD_W_PER_HZ / 1.2589254117941673e-3, 1.0)
        self.assertEqual(FormulationConfig().noise_power, ExperimentConfig().noise_power)
        self.assertEqual(DeviceSpec('d').max_power, ExperimentConfig().max_device_power)
        self.assertEqual(normalize_devices([{'id': 'd'}])['d'].max_power, TX_PSD_W_PER_HZ)

    def test_all_entry_points_share_psd_defaults(self):
        from compared_method.hybrid_topk_location_aware_backhaul import run_chained_pipeline
        from compared_method.hybrid_topk_gssgd_backhaul import run_hybrid_topk_gssgd_backhaul
        from compared_method.WDMoE import _run_wdmoe_pipeline
        from our_alg.similarity_aware_topk_backhaul import run_similarity_aware_topk_backhaul
        for entry in (run_chained_pipeline, run_hybrid_topk_gssgd_backhaul, _run_wdmoe_pipeline, run_similarity_aware_topk_backhaul):
            params = inspect.signature(entry).parameters
            self.assertEqual(params['noise_power'].default, NOISE_PSD_W_PER_HZ)
            self.assertEqual(params['max_device_power'].default, TX_PSD_W_PER_HZ)
        self.assertEqual(inspect.signature(build_simple_devices).parameters['max_power'].default, TX_PSD_W_PER_HZ)

    def test_shannon_linear_sinr_and_small_values(self):
        for sinr in (0.0, 0.1, 1.0, 10.0):
            self.assertAlmostEqual(spectral_efficiency(sinr), math.log2(1 + sinr))
        self.assertGreater(spectral_efficiency(1e-20), 0)
        for invalid in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                spectral_efficiency(invalid)

    def test_invalid_psd_and_split_are_rejected(self):
        for noise in (0.0, -1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                FormulationConfig(noise_power=noise)
        for fraction in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                FormulationConfig(common_power_ratio=fraction)

    def test_channel_uses_power_loss_once_and_normalized_fading(self):
        class FixedRng:
            def __init__(self, shadow_db):
                self.values = iter((shadow_db, math.sqrt(2), 0))
            def gauss(self, mean, std):
                return mean + std * next(self.values)
        for distance in (0, 10, 100, 300, 350, 600):
            for shadow in (0, 10):
                _, amplitude, phase = channel_from_distance(distance, 4, FixedRng(shadow))
                expected_power = 10 ** (-(50 + 15 * math.log10(max(distance, 10)) + shadow) / 10)
                self.assertAlmostEqual(amplitude**2 / expected_power, 1.0)
                self.assertEqual(phase, 0.0)

    def test_channel_fading_has_unit_mean_power(self):
        class ShadowlessRng(random.Random):
            counter = 0
            def gauss(self, mean, std):
                self.counter += 1
                return 0.0 if self.counter % 3 == 1 else super().gauss(mean, std)
        rng = ShadowlessRng(42)
        powers = [channel_from_distance(100, 1, rng)[1]**2 / 1e-8 for _ in range(12000)]
        self.assertAlmostEqual(sum(powers) / len(powers), 1.0, delta=0.03)

    def test_channel_is_reproducible_and_validates_distance(self):
        self.assertEqual(channel_from_distance(350, 1, random.Random(9)), channel_from_distance(350, 4, random.Random(9)))
        for distance in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                channel_from_distance(distance, 1, random.Random(9))

    def test_group_rates_match_hand_calculation(self):
        evaluator, group = self.evaluator()
        common, private = evaluator.group_spectral_efficiencies(group)
        self.assertAlmostEqual(common, math.log2(1 + 18 / 13))
        self.assertAlmostEqual(private['0'], math.log2(1 + 4 / 9))
        self.assertAlmostEqual(private['1'], math.log2(1 + 8 / 5))
        # Doubling all PSDs (including noise) preserves every SINR.
        evaluator.config.noise_power *= 2
        from dataclasses import replace
        evaluator.devices = {k: replace(v, max_power=v.max_power * 2) for k, v in evaluator.devices.items()}
        common2, private2 = evaluator.group_spectral_efficiencies(group)
        self.assertAlmostEqual(common2, common)
        for device in private:
            self.assertAlmostEqual(private2[device], private[device])

    def test_bandwidth_has_hz_units_and_meets_stream_deadline(self):
        evaluator, group = self.evaluator()
        common, private = evaluator.group_spectral_efficiencies(group)
        bandwidth = evaluator.derive_group_bandwidth_for_budget(group, 2)
        self.assertAlmostEqual(bandwidth, max(100 / (2 * common), *(100 / (2 * value) for value in private.values())))
        from dataclasses import replace
        common_rates, private_rates = evaluator.compute_rates([replace(group, bandwidth=bandwidth)])
        durations = [100 / common_rates[('s', 'g')]] + [100 / private_rates[('s', 'g', d)] for d in group.devices]
        self.assertAlmostEqual(max(durations), 2.0)
        doubled = evaluator.compute_rates([replace(group, bandwidth=2 * bandwidth)])
        self.assertAlmostEqual(doubled[0][('s', 'g')], 2 * common_rates[('s', 'g')])

    def test_power_split_is_independent_of_feature_overlap(self):
        evaluator, group = self.evaluator()
        common, private = evaluator.group_spectral_efficiencies(group)
        other, other_group = self.evaluator(features=(('a',), ('b',)))
        self.assertEqual(other.group_spectral_efficiencies(other_group), (common, private))
        other.config.common_power_ratio = 0.2
        changed, _ = other.group_spectral_efficiencies(other_group)
        self.assertAlmostEqual(changed, math.log2(1 + 6 / 25))
        self.assertLess(changed, common)

    def test_zero_payload_and_zero_rate(self):
        evaluator, group = self.evaluator(features=(('a',), ('b',)), common_power_ratio=0.0)
        self.assertGreater(evaluator.derive_group_bandwidth_for_budget(group, 2), 0)
        evaluator, group = self.evaluator(features=(('a',), ('a',)), common_power_ratio=1.0)
        self.assertGreater(evaluator.derive_group_bandwidth_for_budget(group, 2), 0)
        evaluator, group = self.evaluator(common_power_ratio=0.0)
        with self.assertRaises(InfeasibleBandwidthBudget):
            evaluator.derive_group_bandwidth_for_budget(group, 2)
        for budget in (0, -1, float('nan')):
            with self.assertRaises(InfeasibleBandwidthBudget):
                evaluator.derive_group_bandwidth_for_budget(group, budget)
        evaluator, group = self.evaluator(features=((), ()), common_power_ratio=0.0)
        self.assertEqual(evaluator.derive_group_bandwidth_for_budget(group, 2), 0)


class LegacyPhysicalLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshots = [load_legacy(ROOT / path) for path in ('RSMA_MoE/SIoT_RSMA', 'legacy/utils/SIoT_RSMA')]

    def test_both_configs_use_shannon_psds_and_correct_bandwidth(self):
        for modules in self.snapshots:
            config = modules['config']
            self.assertEqual(config.config['BANDWIDTH'], 1_000_000)
            self.assertEqual(config.AWGNoise, NOISE_PSD_W_PER_HZ)
            self.assertEqual(config.TX_POWER_PSD, TX_PSD_W_PER_HZ)
            for sinr in (0, 0.1, 1, 10):
                self.assertAlmostEqual(config.rate_mapping(sinr), math.log2(1 + sinr))
            self.assertEqual(config.payload_per_hz(0, 0), 0)
            self.assertTrue(math.isinf(config.payload_per_hz(1, 0)))
            self.assertTrue(math.isinf(config.required_bandwidth(1, 0)))
            with patch.dict(config.config, {'cmn_ratio': 0.7}):
                with self.assertRaises(ValueError):
                    config.power_split()

    def test_all_legacy_channel_copies_use_amplitude_loss(self):
        for modules in self.snapshots:
            for module_name in ('input_settings', 'input_copy'):
                module = modules[module_name]
                with patch.object(module.np.random, 'normal', side_effect=[10.0, math.sqrt(2), 0]):
                    amplitude = module.SIoT.channel_gain(None, 100)
                self.assertAlmostEqual(amplitude**2 / 1e-9, 1)
            self.assertEqual(modules['input_settings'].SIoT(0).power, TX_PSD_W_PER_HZ)

    def test_legacy_private_rates_use_config_not_data_overlap(self):
        modules = self.snapshots[1]
        a, b = (modules['input_settings'].SIoT(i) for i in (0, 1))
        a.gain = b.gain = 1.0
        a.power = b.power = modules['config'].AWGNoise
        for data_ratio in (0, 0.5, 1):
            rates = modules['phase3'].compute_private_rate_group([a, b], data_ratio)
            self.assertAlmostEqual(rates[0], math.log2(1 + 0.4 / 1.4))
        with patch.dict(modules['config'].config, {'cmn_ratio': 0.2, 'pvt_ratio': 0.8}):
            rates = modules['phase3'].compute_private_rate_group([a, b], 0.5)
            self.assertAlmostEqual(rates[0], math.log2(1 + 0.8 / 1.8))

    def test_legacy_single_bandwidth_uses_bits_and_shannon(self):
        modules = self.snapshots[1]
        device = modules['input_settings'].SIoT(0)
        device.gain = 1.0
        device.power = modules['config'].AWGNoise
        device.locations = ['one']
        expected = modules['config'].config['feature_bits'] / 9.5
        for module_name in ('phase3', 'testing_cost'):
            fn = modules[module_name].compute_single_required_bandwidth
            self.assertAlmostEqual(fn([device], 10), expected)
            self.assertTrue(math.isinf(fn([device], 0.5)))
            device.gain = 0
            self.assertTrue(math.isinf(fn([device], 10)))
            device.gain = 1

    def test_empty_common_or_private_streams_need_no_bandwidth(self):
        modules = self.snapshots[1]
        a, b = (modules['input_settings'].SIoT(i) for i in (0, 1))
        for device in (a, b):
            device.gain = 1
            device.power = modules['config'].AWGNoise
            device.distance = 0
        for common_fraction in (0.0, 0.6, 1.0):
            a.locations = ['a']
            b.locations = ['a'] if common_fraction == 1 else ['b']
            with patch.dict(modules['config'].config, {'cmn_ratio': common_fraction, 'pvt_ratio': 1 - common_fraction}):
                for module_name in ('phase3', 'testing_cost'):
                    output = modules[module_name].compute_rsma_required_bandwidth([a, b], 10)
                    bandwidth = output[0] if isinstance(output, tuple) else output
                    self.assertTrue(math.isfinite(bandwidth))
                    self.assertGreater(bandwidth, 0)


class PipelinePhysicalLayerSmokeTest(unittest.TestCase):
    def test_all_five_methods_use_corrected_model(self):
        from dataclasses import replace
        from experiment_runner import run_single_experiment
        from generate_dags import DAGSpec
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                ExperimentConfig(), auto_random_seed=False, num_runs=1, num_dags=2,
                common_dag_spec=DAGSpec(num_nodes=5, num_branches=2,
                    branch_length_range=(1, 3), depth=4, max_width=3),
                num_experts=4, expert_memory_sizes_mb=(14.16, 17.30, 20.45, 23.59),
                expert_inference_times_ms=(35, 45, 58, 72), num_iot_features=8,
                iot_features_per_node_range=(2, 3), num_edge_servers=3,
                num_iot_devices=12, area_size=300, cell_radius=250,
                iot_features_per_device_range=(3, 5), topk_k=2,
                results_dir=Path(tmp) / 'results', output_dir=Path(tmp) / 'dags',
            )
            results = run_single_experiment(config, verbose=False, print_details=False, show_plot=False)
            self.assertEqual(len(results), 5)
            for name, result in results.items():
                with self.subTest(method=name):
                    self.assertTrue(math.isfinite(result.total_cost))
                    self.assertEqual(result.violations, [])
                    bandwidths = [bw for values in result.rsma_group_bandwidths.values() for bw in values]
                    self.assertTrue(bandwidths)
                    self.assertTrue(all(math.isfinite(bw) and bw > 0 for bw in bandwidths))
                    self.assertTrue(all(math.isfinite(rate) and rate >= 0 for rate in result.evaluation.common_rates.values()))
            files = list(config.results_dir.glob('*.json'))
            self.assertEqual(len(files), 5)
            for path in files:
                payload = json.loads(path.read_text())
                params = payload['network_model']['rsma_parameters']
                self.assertEqual(params['noise_power'], NOISE_PSD_W_PER_HZ)
                self.assertEqual(params['max_device_power'], TX_PSD_W_PER_HZ)
                self.assertEqual(params['noise_power_unit'], 'W/Hz')
                self.assertEqual(params['spectral_efficiency_unit'], 'bit/s/Hz')


if __name__ == '__main__':
    unittest.main()
