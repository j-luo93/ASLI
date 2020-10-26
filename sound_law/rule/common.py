"""This file defines some common sound classes like fricatives and consonants.
"""
from .rule import SoundClass

VGA = SoundClass(['-cons', '+son'], name='VGA')  # Vowels, glides and approximants.
V = VGA.intersect(SoundClass(['+syl']), name='V')
NHV = V.intersect(SoundClass(['-hi']), name='NHV')
SHORT = SoundClass(['-long', '-*olong'])
LONG = SoundClass(['+long', '-*olong'])
OLONG = SoundClass(['+long', '+*olong'])
IJ = VGA.intersect(SoundClass(['-back', '+hi', '-round']), name='IJ')
LV = V.intersect(LONG, name='LV')
OLV = V.intersect(OLONG, name='OLV')
SV = V.intersect(SHORT, name='SV')
C = SoundClass(['-syl'], name='C')
OBS = SoundClass(['+cons', '-son'], name='OBS')
SON = SoundClass(['+cons', '+son'], name='SON')
NASAL = SoundClass(['+nas'])
F = C.intersect(OBS).intersect(SoundClass(['+cont'], name='F'))
N = C.intersect(SON).intersect(NASAL, name='N')
NLV = LV.intersect(NASAL, name='NLV')
BV = V.intersect(SoundClass(['+back']), name='BV')
