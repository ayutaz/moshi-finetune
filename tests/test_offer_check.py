import unittest

from tools.offer_check import (
    HOURS_PER_MONTH,
    KNOWN_GOOD_MULTI_GPU,
    affordable_hours,
    check_offer,
    interconnect_is_known_good,
    true_hourly_rate,
)


class TrueHourlyRateTests(unittest.TestCase):
    """The advertised rate is compute only; the disk bills on top of it."""

    def test_the_offer_that_was_destroyed_before_it_ran(self) -> None:
        # 2026-08-27: advertised US$2.0896/h, billed US$3.3327/h. The difference is
        # 900 GB at US$1.00/GB/month.
        rate = true_hourly_rate(dph_total=2.0896, storage_cost=1.00, disk_gb=900)
        self.assertAlmostEqual(rate, 2.0896 + 900 / HOURS_PER_MONTH, places=6)
        self.assertGreater(rate, 3.32)

    def test_the_replacement_offer_reconciles_with_what_was_billed(self) -> None:
        # Offer 44937484: dph_total 2.3560, US$0.20/GB/month, 500 GB, billed US$2.4936/h.
        # The scratch note recorded dph_total as 2.0014 - that belonged to a different offer
        # in the same search, and this test is what caught it.
        rate = true_hourly_rate(dph_total=2.3560, storage_cost=0.20, disk_gb=500)
        self.assertAlmostEqual(rate, 2.4936, places=2)  # 0.0006 of rounding remains

    def test_a_free_disk_leaves_the_rate_alone(self) -> None:
        self.assertEqual(true_hourly_rate(dph_total=1.5, storage_cost=0.0, disk_gb=900), 1.5)

    def test_negative_figures_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            true_hourly_rate(dph_total=-1.0, storage_cost=0.0, disk_gb=0)


class AffordableHoursTests(unittest.TestCase):
    def test_hours_the_limit_still_buys(self) -> None:
        # Post-run1: US$107.301 spent against the US$112.50 new-run limit.
        self.assertAlmostEqual(
            affordable_hours(spent=107.301, limit=112.5, hourly_rate=2.4936), 2.0847, places=3
        )

    def test_being_past_the_limit_buys_nothing_rather_than_going_negative(self) -> None:
        self.assertEqual(affordable_hours(spent=120.0, limit=112.5, hourly_rate=2.5), 0.0)

    def test_a_zero_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            affordable_hours(spent=0.0, limit=1.0, hourly_rate=0.0)


class InterconnectTests(unittest.TestCase):
    """'A100' names two machines and only one of them has trained this model here."""

    def test_the_variant_that_worked(self) -> None:
        self.assertTrue(interconnect_is_known_good("A100-SXM4-80GB"))
        self.assertTrue(interconnect_is_known_good("A100 SXM4"))

    def test_the_variant_that_hung(self) -> None:
        self.assertFalse(interconnect_is_known_good("A100 80GB PCIe"))
        self.assertFalse(interconnect_is_known_good("A100_PCIE"))

    def test_the_known_good_list_is_what_has_been_observed(self) -> None:
        # Pinned so widening it is a deliberate edit with a test change, not a drift.
        self.assertEqual(KNOWN_GOOD_MULTI_GPU, ("SXM4", "SXM5", "NVLINK"))


class CheckOfferTests(unittest.TestCase):
    SXM4 = {
        "gpu_name": "A100 SXM4",
        "num_gpus": 2,
        "dph_total": 2.0,
        "storage_cost": 0.20,
        "disk_space": 500,
    }

    def test_an_affordable_sxm4_offer_passes_clean(self) -> None:
        v = check_offer(self.SXM4, spent=50.0, limit=112.5, planned_hours=3.0, num_gpus_needed=2)
        self.assertTrue(v.usable)
        self.assertEqual(v.warnings, ())

    def test_the_pcie_offer_warns_even_when_affordable(self) -> None:
        # The US$4.29 case: the budget was fine, the interconnect was not.
        offer = {**self.SXM4, "gpu_name": "A100 80GB PCIe"}
        v = check_offer(offer, spent=50.0, limit=112.5, planned_hours=3.0, num_gpus_needed=2)
        self.assertTrue(v.usable)
        self.assertTrue(any("PCIe" in w or "interconnect" in w for w in v.warnings))

    def test_a_single_gpu_job_does_not_care_about_the_interconnect(self) -> None:
        # 4-1 ran a forward pass on one card and was fine.
        offer = {**self.SXM4, "gpu_name": "A100 80GB PCIe", "num_gpus": 1}
        v = check_offer(offer, spent=50.0, limit=112.5, planned_hours=1.0, num_gpus_needed=1)
        self.assertFalse(any("interconnect" in w for w in v.warnings))

    def test_the_disk_can_make_an_offer_unaffordable(self) -> None:
        # The offer destroyed on 2026-08-27, judged against the budget it actually had.
        offer = {
            "gpu_name": "A100 SXM4",
            "num_gpus": 2,
            "dph_total": 2.0896,
            "storage_cost": 1.00,
            "disk_space": 900,
        }
        v = check_offer(offer, spent=102.812, limit=112.5, planned_hours=3.376, num_gpus_needed=2)
        self.assertFalse(v.usable)
        self.assertTrue(any("buys" in r for r in v.reasons))
        # and the same offer at the advertised rate would have looked fine
        cheap = {**offer, "storage_cost": 0.0}
        self.assertTrue(
            check_offer(
                cheap, spent=102.812, limit=112.5, planned_hours=3.376, num_gpus_needed=2
            ).usable
        )

    def test_a_hidden_disk_charge_warns_even_when_the_run_still_fits(self) -> None:
        offer = {**self.SXM4, "storage_cost": 1.00, "disk_space": 900}
        v = check_offer(offer, spent=0.0, limit=112.5, planned_hours=1.0, num_gpus_needed=2)
        self.assertTrue(v.usable)
        self.assertTrue(any("advertised" in w for w in v.warnings))

    def test_too_few_gpus_is_a_refusal(self) -> None:
        offer = {**self.SXM4, "num_gpus": 1}
        v = check_offer(offer, spent=0.0, limit=112.5, planned_hours=1.0, num_gpus_needed=2)
        self.assertFalse(v.usable)


if __name__ == "__main__":
    unittest.main()
