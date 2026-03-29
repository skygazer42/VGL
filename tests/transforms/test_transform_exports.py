from vgl.transforms import (
    BaseTransform,
    Compose,
    IdentityTransform,
    RandomLinkSplit,
    RandomNodeSplit,
)


def test_transform_surface_reexports_base_and_composition_types():
    assert BaseTransform.__name__ == "BaseTransform"
    assert Compose.__name__ == "Compose"
    assert IdentityTransform.__name__ == "IdentityTransform"


def test_random_split_transforms_conform_to_base_transform_contract():
    assert issubclass(RandomNodeSplit, BaseTransform)
    assert issubclass(RandomLinkSplit, BaseTransform)


def test_legacy_data_transform_module_reexports_current_transform_surface():
    from vgl.data.transform import AddSelfLoops as LegacyAddSelfLoops
    from vgl.data.transform import BaseTransform as LegacyBaseTransform
    from vgl.data.transform import Compose as LegacyCompose
    from vgl.data.transform import FeatureStandardize as LegacyFeatureStandardize
    from vgl.data.transform import IdentityTransform as LegacyIdentityTransform
    from vgl.data.transform import LargestConnectedComponents as LegacyLargestConnectedComponents
    from vgl.data.transform import NormalizeFeatures as LegacyNormalizeFeatures
    from vgl.data.transform import RandomGraphSplit as LegacyRandomGraphSplit
    from vgl.data.transform import RandomLinkSplit as LegacyRandomLinkSplit
    from vgl.data.transform import RandomNodeSplit as LegacyRandomNodeSplit
    from vgl.data.transform import RemoveSelfLoops as LegacyRemoveSelfLoops
    from vgl.data.transform import ToUndirected as LegacyToUndirected
    from vgl.data.transform import TrainOnlyFeatureNormalizer as LegacyTrainOnlyFeatureNormalizer
    from vgl.transforms import AddSelfLoops
    from vgl.transforms import FeatureStandardize
    from vgl.transforms import LargestConnectedComponents
    from vgl.transforms import NormalizeFeatures
    from vgl.transforms import RandomGraphSplit
    from vgl.transforms import RemoveSelfLoops
    from vgl.transforms import ToUndirected
    from vgl.transforms import TrainOnlyFeatureNormalizer

    assert LegacyAddSelfLoops is AddSelfLoops
    assert LegacyBaseTransform is BaseTransform
    assert LegacyCompose is Compose
    assert LegacyFeatureStandardize is FeatureStandardize
    assert LegacyIdentityTransform is IdentityTransform
    assert LegacyLargestConnectedComponents is LargestConnectedComponents
    assert LegacyNormalizeFeatures is NormalizeFeatures
    assert LegacyRandomGraphSplit is RandomGraphSplit
    assert LegacyRandomLinkSplit is RandomLinkSplit
    assert LegacyRandomNodeSplit is RandomNodeSplit
    assert LegacyRemoveSelfLoops is RemoveSelfLoops
    assert LegacyToUndirected is ToUndirected
    assert LegacyTrainOnlyFeatureNormalizer is TrainOnlyFeatureNormalizer
