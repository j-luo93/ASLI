"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sound_law.data.alphabet import Alphabet


class SoundChangeActionSpace:
    """The action space, i.e., the space of all sound changes."""

    def __init__(self, abc: Alphabet):
        self.abc = abc

    def __len__(self):
        # HACK(j_luo) expand this later
        return 10

    def get_action(self, idx: int) -> SoundChangeAction:
        # FIXME(j_luo) api is weird.
        return SoundChangeAction(self, idx)


@dataclass
class SoundChangeAction:
    """One sound change rule."""

    space: SoundChangeActionSpace = field(repr=False)
    action_id: int
    before: str = field(init=False)
    after: str = field(init=False)
    before_id: int = field(init=False)
    after_id: int = field(init=False)

    def __post_init__(self):
        # HACK(j_luo) expand this later
        idx2before_after = {
            0: ('l', 'b'),
            1: ('l', 'e'),
            2: ('b', 'l'),
            3: ('e', 'i'),
            4: ('e', 'd'),
            5: ('d', 'e'),
            6: ('l', 'e'),
            7: ('l', 's'),
            8: ('s', 'l'),
            9: ('t', 'l')
        }
        self.before, self.after = idx2before_after[self.action_id]

        abc = self.space.abc
        self.before_id = abc[self.before]
        self.after_id = abc[self.after]

    def __str__(self):
        return f'{self.before} > {self.after}'
