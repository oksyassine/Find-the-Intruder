# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

class UserProfile:
    def __init__(self, word: str = None, response: str = None, score: int = 0):
        self.word = word
        self.response = response
        self.score = score
