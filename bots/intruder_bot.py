# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import (
    ActivityHandler,
    ConversationState,
    TurnContext,
    UserState,
    CardFactory,
    MessageFactory,
)
from botbuilder.schema import (
    ChannelAccount,
    HeroCard,
    CardAction,
    ActivityTypes,
    Attachment,
    AttachmentData,
    Activity,
    ActionTypes,
)
from data_models import ConversationFlow, Question, UserProfile
import random
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('embedneg10')
vocab=list(model.vocab.keys())
intrus=""

class ValidationResult:
    def __init__(
        self, is_valid: bool = False, value: object = None, message: str = None
    ):
        self.is_valid = is_valid
        self.value = value
        self.message = message

class IntruderBot(ActivityHandler):
    def __init__(self, conversation_state: ConversationState, user_state: UserState):
        if conversation_state is None:
            raise TypeError(
                "[CustomPromptBot]: Missing parameter. conversation_state is required but None was given"
            )
        if user_state is None:
            raise TypeError(
                "[CustomPromptBot]: Missing parameter. user_state is required but None was given"
            )

        self.conversation_state = conversation_state
        self.user_state = user_state

        self.flow_accessor = self.conversation_state.create_property("ConversationFlow")
        self.profile_accessor = self.user_state.create_property("UserProfile")

    async def on_message_activity(self, turn_context: TurnContext):
        # Get the state properties from the turn context.
        profile = await self.profile_accessor.get(turn_context, UserProfile)
        flow = await self.flow_accessor.get(turn_context, ConversationFlow)

        if turn_context.activity.text == "start":
            return await turn_context.send_activities([
                Activity(
                    type=ActivityTypes.typing
                    ),
                Activity(
                    type="delay",
                    value=2000
                    ),
                Activity(
                    type=ActivityTypes.message,
                    text="Welcome ! Thank you for subscribing.\r\nThis bot will present you a list of words and you should find the intruder.\r\nIf you ever want to test your lexical skills, just type a significant word and Let's Start !"
                    )
                ])
        elif turn_context.activity.text == "help":
            return await turn_context.send_activity(
                MessageFactory.text("Intruder Bot\r\nUsage: type anything to start\r\nThank you!"
                )
            )
        else:
            await self._fill_out_user_profile(flow, profile, turn_context)
            # Save changes to UserState and ConversationState
            await self.conversation_state.save_changes(turn_context)
            await self.user_state.save_changes(turn_context)

    async def _fill_out_user_profile(
        self, flow: ConversationFlow, profile: UserProfile, turn_context: TurnContext
    ):
        user_input = turn_context.activity.text.strip()

        # ask for name
        if flow.last_question_asked == Question.NONE:
            await turn_context.send_activity(
                MessageFactory.text("Let's get started. Please type a significant word !")
            )
            flow.last_question_asked = Question.WORD

        # validate name then ask for age
        elif flow.last_question_asked == Question.WORD:
            validate_result = self._validate_word(user_input)
            if not validate_result.is_valid:
                await turn_context.send_activity(
                    MessageFactory.text(validate_result.message)
                )
            else:
                profile.word = validate_result.value
                await turn_context.send_activity(
                    MessageFactory.text(f"You choose: {profile.word}")
                )
                global intrus
                most=model.most_similar(profile.word)
                liste=[]
                for i in range(5):
                    liste.append(most[:5][i][0])
                sim=1
                while (sim>0.5 or sim<0.3):
                    intrus=vocab[random.randint(1,111000)]
                    sim=model.similarity(profile.word,intrus)
                liste.insert(random.randrange(len(liste)),intrus)

                card = HeroCard(
                    text="Here is the list of the words.\r\nPlease choose the intruder one!",
                    buttons=[
                        CardAction(
                            type=ActionTypes.im_back, title=liste[0], value=liste[0]
                        ),
                        CardAction(
                            type=ActionTypes.im_back, title=liste[1], value=liste[1]
                        ),
                        CardAction(
                            type=ActionTypes.im_back, title=liste[2], value=liste[2]
                        ),
                        CardAction(
                            type=ActionTypes.im_back, title=liste[3], value=liste[3]
                        ),
                        CardAction(
                            type=ActionTypes.im_back, title=liste[4], value=liste[4]
                        ),
                        CardAction(
                            type=ActionTypes.im_back, title=liste[5], value=liste[5]
                        ),
                    ],
                )

                reply = MessageFactory.attachment(CardFactory.hero_card(card))

                await turn_context.send_activity(reply)
                flow.last_question_asked = Question.RESPONSE

        # validate date and wrap it up
        elif flow.last_question_asked == Question.RESPONSE:
            if user_input == "exit":
                await turn_context.send_activity(
                    MessageFactory.text("Type anything to run the Intruder Bot again.")
                )
                flow.last_question_asked = Question.NONE
            else:
                validate_result = self._validate_result(user_input)
                if not validate_result.is_valid:
                    card = HeroCard(
                        text=validate_result.message,
                        buttons=[
                            CardAction(
                                type=ActionTypes.im_back, title="exit", value="exit"
                            ),
                        ],
                    )
                    reply = MessageFactory.attachment(CardFactory.hero_card(card))
                    await turn_context.send_activity(reply)
                else:
                    profile.response = validate_result.value
                    profile.score += 1
                    await turn_context.send_activity(
                        MessageFactory.text(
                            f"You have responded correctly.\r\nYour score is: {profile.score}"
                        )
                    )
                    await turn_context.send_activity(
                        MessageFactory.text(
                            f"Thanks for completing the test."
                        )
                    )
                    await turn_context.send_activity(
                        MessageFactory.text("Type anything to run the Intruder Bot again.")
                    )
                    flow.last_question_asked = Question.NONE

    def _validate_word(self, user_input: str) -> ValidationResult:
        word=user_input.lower()
        tst=False
        for w in vocab:
            if word == w:
                tst=True
        if " " in word:
            return ValidationResult(
                is_valid=False,
                message="Please type just one word",
            )
        elif not tst:
            return ValidationResult(
                is_valid=False,
                message="Please choose a significant word",
            )
        return ValidationResult(is_valid=True, value=user_input)

    def _validate_result(self, user_input: str) -> ValidationResult:
        tst = False
        if intrus == user_input:
            tst=True
        elif not tst:
            return ValidationResult(
                is_valid=False, message="Your response is incorrect.\r\nTry again, or click on cancel if you want to exit the test."
            )
        return ValidationResult(is_valid=True, value=user_input)
