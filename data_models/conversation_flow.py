# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from enum import Enum

class Question(Enum):
    WORD = 1
    RESPONSE = 2
    NONE = 3

class ConversationFlow:
    def __init__(
        self, last_question_asked: Question = Question.NONE,
    ):
        self.last_question_asked = last_question_asked
