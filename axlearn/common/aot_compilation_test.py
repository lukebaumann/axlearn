"""Tests aot_compilation utils."""

from typing import cast
from unittest import mock

from absl.testing import absltest


from axlearn.common import test_utils
from axlearn.common.aot_compilation import get_devices_for_topology, reshape_devices
from axlearn.common.utils import HybridMeshShape


class FakeDevice:
    def __init__(
        self,
        platform="tpu",
        device_kind="TPU v4",
        process_index=0,
        coords=(0, 0, 0),
        slice_index=0,
    ):
        self.platform = platform
        self.device_kind = device_kind
        self.process_index = process_index
        self.coords = coords
        self.slice_index = slice_index


class AOTCompilationTest(test_utils.TestCase):
    def test_reshape_devices(self):
        devices = [FakeDevice()] * 8
        mesh_shape = (-1, 2)
        devices_per_slice = 8
        num_slices = 1

        devices, mesh_shape = reshape_devices(
            devices=devices,
            mesh_shape=mesh_shape,
            devices_per_slice=devices_per_slice,
            num_slices=num_slices,
        )
        self.assertEqual(mesh_shape, (4, 2))
        self.assertEqual(devices.shape, mesh_shape)

        devices = [FakeDevice()] * 8
        mesh_shape = HybridMeshShape(ici_mesh_shape=(1, -1), dcn_mesh_shape=(-1, 1))
        devices_per_slice = 4
        num_slices = 2
        devices, mesh_shape = reshape_devices(
            devices=devices,
            mesh_shape=mesh_shape,
            devices_per_slice=devices_per_slice,
            num_slices=num_slices,
        )
        self.assertIsInstance(mesh_shape, HybridMeshShape)
        mesh_shape = cast(HybridMeshShape, mesh_shape)  # Make pytype happy.
        self.assertEqual(mesh_shape.ici_mesh_shape, (1, 4))
        self.assertEqual(mesh_shape.dcn_mesh_shape, (2, 1))
        self.assertEqual(devices.shape, (2, 4))

    @mock.patch("axlearn.common.aot_compilation.get_topology_desc")
    @mock.patch("jax.devices")
    def test_get_devices_for_topology_pathways(self, mock_jax_devices, mock_get_topology_desc):
        mock_pathwaysutils = mock.MagicMock()
        mock_pathwaysutils.is_pathways_backend_used.return_value = True

        with mock.patch.dict("sys.modules", {"pathwaysutils": mock_pathwaysutils}):
            fake_coords = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
            fake_real_devices = [
                FakeDevice(
                    platform="proxy",
                    device_kind="TPU v4",
                    process_index=0,
                    coords=fake_coords[i],
                    slice_index=0,
                )
                for i in range(4)
            ]
            mock_jax_devices.return_value = fake_real_devices

            mock_topology = mock.MagicMock()
            fake_compile_devices = [
                FakeDevice(
                    platform="tpu",
                    device_kind="TPU v4",
                    process_index=0,
                    coords=fake_coords[i],
                    slice_index=0,
                )
                for i in range(4)
            ]
            mock_topology.devices = fake_compile_devices
            mock_get_topology_desc.return_value = mock_topology

            devices, devices_per_slice = get_devices_for_topology("v4-8", topology_num_slices=1)

            self.assertEqual(devices, fake_real_devices)
            self.assertEqual(devices_per_slice, 4)
            mock_get_topology_desc.assert_called_once_with(
                platform="tpu",
                topology_name="v4:2x2x1",
                chip_config_name="megacore",
                chips_per_host_bounds=(2, 2, 1),
                num_slices=1,
            )

    @mock.patch("axlearn.common.aot_compilation.get_topology_desc")
    @mock.patch("jax.devices")
    def test_get_devices_for_topology_non_pathways(self, mock_jax_devices, mock_get_topology_desc):
        mock_pathwaysutils = mock.MagicMock()
        mock_pathwaysutils.is_pathways_backend_used.return_value = False

        with mock.patch.dict("sys.modules", {"pathwaysutils": mock_pathwaysutils}):
            mock_get_topology = mock.MagicMock()
            fake_topology_devices = [FakeDevice()] * 4
            mock_get_topology.devices = fake_topology_devices
            mock_get_topology_desc.return_value = mock_get_topology

            devices, devices_per_slice = get_devices_for_topology("v4-8", topology_num_slices=1)

            self.assertEqual(devices, fake_topology_devices)
            self.assertEqual(devices_per_slice, 4)
            mock_get_topology_desc.assert_called_once()
            mock_jax_devices.assert_not_called()

    @mock.patch("axlearn.common.aot_compilation.get_topology_desc")
    @mock.patch("jax.devices")
    def test_get_devices_for_topology_no_pathwaysutils(self, mock_jax_devices, mock_get_topology_desc):
        with mock.patch.dict("sys.modules", {"pathwaysutils": None}):
            mock_get_topology = mock.MagicMock()
            fake_topology_devices = [FakeDevice()] * 4
            mock_get_topology.devices = fake_topology_devices
            mock_get_topology_desc.return_value = mock_get_topology

            devices, devices_per_slice = get_devices_for_topology("v4-8", topology_num_slices=1)

            self.assertEqual(devices, fake_topology_devices)
            self.assertEqual(devices_per_slice, 4)
            mock_get_topology_desc.assert_called_once()
            mock_jax_devices.assert_not_called()


if __name__ == "__main__":
    absltest.main()
