"""Check the real compiled Push-T interfaces and recorded contact identity on CPU."""

import numpy as np
import pytest

from shakebench.environments.push_t import PushT


def test_mu050_is_effective_on_table_pairs_with_pusher_and_global_profile_preserved():
    env = PushT(physics_profile="official")
    try:
        model = env.sim.model._model
        table_pairs = []
        pusher_pairs = []
        for index, (first, second) in enumerate(zip(model.pair_geom1, model.pair_geom2)):
            ids = {int(first), int(second)}
            if not ids.intersection(env.tee_geom_ids):
                continue
            (table_pairs if env.table_geom_id in ids else pusher_pairs).append(index)
        assert len(table_pairs) == len(env.tee_geom_ids)
        assert pusher_pairs
        for indices in (table_pairs, pusher_pairs):
            np.testing.assert_allclose(model.pair_friction[indices, :2], 0.50, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                model.pair_friction[indices, 2:],
                np.tile([0.005, 0.0001, 0.0001], (len(indices), 1)),
                rtol=0,
                atol=1e-12,
            )
        assert env.physics_profile.contact["sliding_mu"]["table_object"] == pytest.approx(0.30)
        context = env.get_policy_task_context()
        assert context["table_sliding_mu"] == pytest.approx(0.50)
        assert context["pusher_sliding_mu"] == pytest.approx(0.50)
        assert context["contact_physics_revision"] == "push_t_contact_v2"
        assert context["version"] == 5
        assert context["success_hold_s"] == pytest.approx(0.5)
    finally:
        env.close()
