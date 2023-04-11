# pylint: disable=missing-function-docstring,missing-class-docstring
from .. import _constraint as constraint


class TestRange:
    def test_repr(self):
        r = constraint.Range(ge=[1], lt=[2])
        assert repr(r) == "Range(>=[1], <[2])"

        r = constraint.Range(gt=[1], le=[2])
        assert repr(r) == "Range(>[1], <=[2])"

        r = constraint.Range(gt=[1], ge=[1], lt=[3], le=[2])
        assert repr(r) == "Range(>=[1], >[1], <=[2], <[3])"
