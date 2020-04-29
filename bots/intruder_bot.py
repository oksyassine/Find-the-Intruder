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
    HeroCard,
    CardAction,
    ActivityTypes,
    Attachment,
    Activity,
    ActionTypes,
    SuggestedActions,
)
from data_models import ConversationFlow, Question, UserProfile
# import numpy as np
# import math
# from scipy.spatial.distance import cosine
# with open('embed30.txt') as f:
#     model = dict()
#     for line in f.readlines()[1:]:
#         row = line.split()
#         word = row[0]
#         vector = np.array([float(x) for x in row[1:]])
#         model[word] = vector
#
# vocab=[w for w in model]
#
# def distance(w1, w2):
#     return cosine(model[w1],model[w2])
#
# def closest_words(word):
#     distances = [
#             (w, cosine(model[word], model[w]))
#             for w in model
#     ]
#     closest = sorted(distances, key=lambda item: item[1])[1:6]
#     return [w for w,_ in closest]
try:
    with open('api_key') as f:
        api_key=f.readline().strip()
except FileNotFoundError as e:
    print("File Not Found: api_key")
import random,os,json
from datetime import datetime
from PyDictionary import PyDictionary
dic=PyDictionary()
#load our model
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('embed30.txt')
vocab=list(model.vocab.keys())

intrus=""
liste=[]
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
        text=turn_context.activity.text.lower()
        if text == "get started" or text == "/start":
            return await turn_context.send_activities([
                Activity(
                    type=ActivityTypes.typing
                    ),
                Activity(
                    type="delay",
                    value=1000
                    ),
                Activity(
                    type=ActivityTypes.message,
                    text="Welcome ! Thank you for subscribing.\n\nThis bot will present you a list of words similar to the word that you will type, and you should find the intruder one.\n\nYou can use the built-in dictionary at anytime to search the meanings of a word.\n\nIf you want any help just type 'help me' .\n\nIf you ever want to test your lexical skills, just type a significant word and Let's Start !"
                    )
                ])
        elif text == "help me" or text == "/help":
            return await turn_context.send_activity(
                MessageFactory.text("Intruder Bot\n\nUsage: \n\n _Def word_ You can use the built-in dictionary at anytime to search the meanings of a word.\n\nType anything to start the test, then when the conversation starts, you will be asked to type a word, and we will send you 5 most similar words to that word and an intruder one that you have to detect in order to pass the test. \n\n Your score will be incremented on each correct response, and will be decremented on each incorrect response.\n\n Let's Play !"
                )
            )
        elif "def" == text.split()[0]:
            if len(text.split()) < 2:
                return await turn_context.send_activity(
                    MessageFactory.text("Please type a word to define !")
                )
            elif len(text.split()) < 3:
                word = text.split()[1]
                meaning=dic.meaning(word)
                if meaning:
                    df=list(meaning.items())
                    for it in df:
                        await turn_context.send_activity(
                            MessageFactory.text(f"**{it[0]}** : \n\n* "+ " \n\n* ".join(it[1]))
                        )
                else:
                    #wordnik_api
                    return await turn_context.send_activity(
                        MessageFactory.text("Please type a significant word !")
                    )
            else:
                return await turn_context.send_activity(
                    MessageFactory.text("Please type just one word !")
                )
        elif text == "word of the day" or text == "/word_day":
            today=datetime.today().strftime('%Y-%m-%d')
            getwordofday = f"curl -X GET --header 'Accept: application/json' 'https://api.wordnik.com/v4/words.json/wordOfTheDay?date={today}&api_key={api_key}'"
            result=os.popen(getwordofday).read()
            res=json.loads(result)
            await turn_context.send_activity(
                MessageFactory.text(f"The word of today is : **{res['word']}**")
                )
            for it in res['definitions']:
                await turn_context.send_activity(
                    MessageFactory.text(f"**{it['partOfSpeech']}** : \n\n* "+it['text'])
                )
            examples=[(example['title'],example['text']) for example in res['examples']]
            for it in range(len(examples)):
                await turn_context.send_activity(
                    MessageFactory.text(f"_Example{it+1}_ \n\n**{examples[it][0]}** : \n\n* "+examples[it][1])
                )
            if res['note']:
                await turn_context.send_activity(
                    MessageFactory.text("**Note**"+" : \n\n* "+res['note'])
                )
        else:
            await self._fill_out_user_profile(flow, profile, turn_context)
            # Save changes to UserState and ConversationState
            await self.conversation_state.save_changes(turn_context)
            await self.user_state.save_changes(turn_context)

    async def _fill_out_user_profile(
        self, flow: ConversationFlow, profile: UserProfile, turn_context: TurnContext
    ):
        user_input = turn_context.activity.text.strip().lower()

        #begin a conversation and ask for a significant word
        if flow.last_question_asked == Question.NONE:
            await turn_context.send_activity(
                MessageFactory.text("Let's get started. Please type a significant word !")
            )
            flow.last_question_asked = Question.WORD

        #validate word and ask for the response
        elif flow.last_question_asked == Question.WORD:
            validate_result = self._validate_word(user_input)
            if not validate_result.is_valid:
                await turn_context.send_activity(
                    MessageFactory.text(validate_result.message)
                )
            else:
                profile.word = validate_result.value
                await turn_context.send_activity(
                    MessageFactory.text(f"You choose: {turn_context.activity.text.strip()} \n\nPlease Wait.")
                )
                await turn_context.send_activities([
                    Activity(
                        type=ActivityTypes.typing
                        ),
                    Activity(
                        type="delay",
                        value=2000
                        )])
                global intrus
                global liste
                #liste=closest_words(profile.word)
                most=model.most_similar(profile.word)
                liste=[]
                for i in range(5):
                    liste.append(most[:5][i][0])
                sim=1
                while sim>0.5:
                    intrus=vocab[random.randint(1,len(vocab)-1)]
                    if intrus in liste:
                        continue
                    meaning=dic.meaning(intrus)
                    if not meaning:
                        #wordnik api
                        continue
                    #sim=distance(profile.word,intrus)
                    sim=model.similarity(profile.word,intrus)
                liste.insert(random.randrange(len(liste)),intrus)
                card = HeroCard(
                    text="Here is the list of the words.\n\nPlease choose the intruder one!",
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

        # validate response and wrap it up
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
                    profile.score -= 1
                    await turn_context.send_activity(MessageFactory.text(f"Your Score is: {profile.score}"))
                    suggested = MessageFactory.text("Chase the Intruder!")
                    suggested.suggested_actions = SuggestedActions(
                        actions=[
                            CardAction(title=liste[0], type=ActionTypes.im_back, value=liste[0]),
                            CardAction(title=liste[1], type=ActionTypes.im_back, value=liste[1]),
                            CardAction(title=liste[2], type=ActionTypes.im_back, value=liste[2]),
                            CardAction(title=liste[3], type=ActionTypes.im_back, value=liste[3]),
                            CardAction(title=liste[4], type=ActionTypes.im_back, value=liste[4]),
                            CardAction(title=liste[5], type=ActionTypes.im_back, value=liste[5]),
                        ]
                    )
                    await turn_context.send_activity(suggested)
                else:
                    profile.response = validate_result.value
                    profile.score += 1
                    await turn_context.send_activity(
                        MessageFactory.text(
                            f"You have responded correctly.\n\nYour score is: {profile.score}\n\nThanks for completing the test."
                        )
                    )
                    await turn_context.send_activity(
                        MessageFactory.text("Type anything to run the Intruder Bot again.")
                    )
                    flow.last_question_asked = Question.NONE

    def _validate_word(self, user_input: str) -> ValidationResult:
        if " " in user_input:
            return ValidationResult(
                is_valid=False,
                message="Please type just one word",
            )
        tst=False
        for w in vocab:
            if user_input == w:
                tst=True
                meaning=dic.meaning(user_input)
                if not meaning:
                    tst=False
                    #wordnik api
        if not tst:
            return ValidationResult(
                is_valid=False,
                message="Please type a significant word",
            )
        return ValidationResult(is_valid=True, value=user_input)

    def _validate_result(self, user_input: str) -> ValidationResult:
        tst = False
        if intrus == user_input:
            tst=True
        elif not tst:
            return ValidationResult(
                is_valid=False, message="Your response is incorrect.\n\nTry again, or click on exit if you want to abort the test."
            )
        return ValidationResult(is_valid=True, value=user_input)
