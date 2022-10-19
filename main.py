import spacy
from spacy.tokens import Doc as SpacyDoc

"""
Author Anatole Gershman
Spacy English parser produces a Spacy document
This code transforms SpacyDocs into UdDocs
Usage:
If you already have a Spacy document (output of Spacy), use this to create a UdDoc:
ud_doc = spacy_to_ud_doc(spacy_doc)

to print a UdDoc, use:
ud_doc.print_doc()

To create a Spacy doc, first create an instance of SpacyParser
spr = SpacyParser()

To get a Spacy doc, use
spacy_doc = spr.nlp('Spacy parser is pretty good')

To print a Spacy doc, use:
print_spacy_doc(spacy_doc)
"""


class UdDoc:

    def __init__(self):
        self.sentences = []

    def print_doc(self, fo=None, count=0):
        """
        :param fo: for printing to a file
        :param count: would put a number on a sentence
        :return: nothing
        """
        s: SentenceNode
        for s in self.sentences:
            txt = s.text
            if count > 0:
                txt = f"\n {count} {txt}"
            if fo:
                fo.write(txt + '\n')
            else:
                print(txt)
            s.print_words(fo=fo)
            if not fo:
                print()


class SentenceNode:
    """ A sentence is a list of words """

    def __init__(self, doc: UdDoc):
        self.word_nodes = []  # list of word nodes - actual nodes, not names
        self.text = ""    # the original text - a string
        self.doc = doc      # document object

    def print_words(self, fo=None):
        w: WordNode
        for w in self.word_nodes:
            w.display_word(fo=fo)


class WordNode:
    wn_count = 0
    all_wn = []

    def __init__(self, index, text, sentence_node):
        self.wn_id = WordNode.wn_count
        WordNode.wn_count += 1
        WordNode.all_wn.append(self)
        self.index = index   # in the sentence 1-based self.start_index = self.index
        self.index_span = [self.index, self.index]
        self.text = text  # string, word text
        self.sentence_node = sentence_node  # the parent sentence instance
        self.lemma = None
        self.upos = None     # universal pos tag
        self.features = {}  # {feature-name: feature-value}
        self.span = None     # word span in characters, [start_char, end_char] with respect to the document
        # if the word node is the head of the NER from Spacy, the NER info is recorded
        self.ner = None  # {text: phrase_text, words: [words], span: phrase_span, type: phrase_type}
        self.ner_head_word = None
        # UD tree has dependency edges from dependent nodes to their governors labeled with the dependency relation
        # these come from the underlying UD parser (e.g., Stanza)
        self.dependency_relation = None  # such as 'nsubj', 'obj', etc
        self.governor = None  # 1-based index in sentence
        self.governor_word = None  # set by add_dominates

    def display_word(self, fo=None):
        res = f"{self.index}\t{self.text}\tlemma: {self.lemma}\tpos: {self.upos}"
        res = res + f"\tdep: {self.dependency_relation}\tgov: {self.governor}"
        feats = ""
        for f_name, f_val in self.features.items():
            feats += f"{f_name}={f_val}|"
        feats = feats.strip('|')
        if not feats:
            feats = 'None'
        res += f"\tfeats: {feats}"
        if self.ner:
            word_indices = [x.index for x in self.ner.get('words')]
            res += f"\tNER-type: {self.ner.get('type')}\tNER-words: {word_indices}"
        if fo:
            fo.write(res + '\n')
        else:
            print(res)


class SpacyParser:

    def __init__(self, lang='en', model_name='en_core_web_trf'):
        self.language = lang
        self.nlp = spacy.load(model_name)


def print_spacy_doc(sp_doc: spacy.tokens.Doc):
    """  Spacy token indexes are 0-based, UD indexes are 1-based"""
    print("Spacy tokens for: " + sp_doc.text)
    for token in sp_doc:
        feats = morph_to_string(token)
        line = f"{token.i}\t{token.text}\tlemma: {token.lemma_}\tpos: {token.pos_}\tdep: {token.dep_}"
        line += f"\tgov: {token.head.i}\tfeats: {feats}"
        print(line)
    if sp_doc.ents:
        print("Entities:")
        for ent in sp_doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)


def morph_to_string(token):
    f_dict = token.morph.to_dict()
    f_str = ""
    for f_name, f_val in f_dict.items():
        f_str += f"{f_name}={f_val}|"
    f_str = f_str.strip('|')
    if not f_str:
        f_str = 'None'
    return f_str


def spacy_to_ud_doc(sp_doc: SpacyDoc):
    doc = UdDoc()
    spacy_sentence: SpacyDoc
    spacy_sentences = [x for x in sp_doc.sents]
    if not spacy_sentences:
        # if spacy_doc is not divided into sentences then the whole doc is the sentence
        spacy_sentences = [sp_doc]
    for spacy_sentence in spacy_sentences:
        spacy_to_ud_sentence(spacy_sentence, doc)
    return doc


def spacy_to_ud_sentence(spacy_sent: SpacyDoc, doc: UdDoc):
    ud_sent = SentenceNode(doc)
    ud_sent.text = spacy_sent.text
    spacy_token: spacy.tokens.Token
    for spacy_token in spacy_sent:
        new_word = WordNode(spacy_token.i + 1, spacy_token.text, ud_sent)
        spacy_to_ud_token(new_word, spacy_token)  # keeps some Spacy dependencies
        ud_sent.word_nodes.append(new_word)  # transforms dependencies
    add_governor_word(ud_sent)
    spacy_to_ud(ud_sent)
    add_entities(ud_sent, spacy_sent)
    doc.sentences.append(ud_sent)


def spacy_to_ud_token(word_node: WordNode, spacy_token: spacy.tokens.Token):
    word_node.upos = spacy_token.pos_
    word_node.lemma = spacy_token.lemma_
    s_dep = spacy_token.dep_
    dep_dict = {'dobj': 'obj', 'dative': 'iobj', 'nsubjpass': 'nsubj:pass', 'csubjpass': 'csubj:pass', 'ROOT': 'root',
                'auxpass': 'aux:pass', 'preconj': 'pre:conj', 'prt': 'compound:prt', 'predet': 'det:predet',
                'poss': 'nmod:poss', 'relcl': 'acl:relcl', 'neg': 'advmod', 'quantmod': 'compound',
                'parataxis': 'prataxis'}
    dep = dep_dict.get(s_dep, s_dep)
    word_node.dependency_relation = dep
    spacy_head: spacy.tokens.Token
    spacy_head = spacy_token.head
    if spacy_head == spacy_token:
        word_node.governor = 0
    else:
        word_node.governor = spacy_head.i + 1
    start_char = spacy_token.idx
    end_char = start_char + len(spacy_token.text)
    word_node.span = [start_char, end_char]
    word_node.features = spacy_token.morph.to_dict()


def add_governor_word(sent: SentenceNode):
    # add governor word to each word
    word_node: WordNode
    for word_node in sent.word_nodes:
        gov_ind = word_node.governor
        if gov_ind > 0:
            gov_word = sent.word_nodes[gov_ind - 1]
            word_node.governor_word = gov_word


"""
UD makes a distinction between core and non-core verbal clause complements
For example, in "I said that Mary ate an apple" said <-ccomp- ate
because we expect a clause expressing what was said
ccomp is a core clausal complement
advcl - adverbial clause is for non-core (not necessarily expected) clausal complements
For example: "He worked so that Mary could eat"  worked <-advcl- eat
When the implied subject of the clausal complement is the same as the subject of the main clause, UD uses xcomp:
"He worked to eat", although Stanza produces xcomp while Spacy produces advcl
"""


def spacy_to_ud(sent: SentenceNode):
    """ we need to convert Spacy dependencies to Universal dependencies
    """
    word_node: WordNode
    prep_node: WordNode
    verb_node: WordNode
    for word_node in sent.word_nodes:
        if word_node.dependency_relation == 'aux':
            aux_to_ud(word_node)
        elif word_node.dependency_relation == 'oprd':
            oprd_to_ud(word_node)
        elif word_node.dependency_relation == 'amod':
            amod_to_ud(word_node)
        elif word_node.dependency_relation == 'nmod':
            nmod_to_ud(word_node)
        elif word_node.dependency_relation == 'nummod':
            nummod_to_ud(word_node)
        elif word_node.dependency_relation == 'advcl':
            advcl_to_ud(word_node)
        elif word_node.dependency_relation == 'pobj':
            pobj_to_ud(word_node)
        elif word_node.dependency_relation == 'pcomp':
            pcomp_to_ud(word_node)
        elif word_node.dependency_relation in ['xcomp', 'ccomp']:
            comp_to_ud(word_node)
        elif word_node.dependency_relation == 'attr':
            attr_to_ud(word_node)
        elif word_node.dependency_relation == 'acomp':
            acomp_to_ud(word_node)
        elif word_node.dependency_relation == 'advmod':
            advmod_to_ud(word_node)
        elif word_node.dependency_relation == 'npadvmod':
            npadvmod_to_ud(word_node)
        elif word_node.dependency_relation == 'conj':
            conj_to_ud(word_node)
        elif word_node.dependency_relation == 'dep':
            dep_to_ud(word_node)
    fix_advmod_cop(sent)


def aux_to_ud(word_node: WordNode):
    """ way to get ... PART (to) -aux-> VERB (get)  ==> PART (to) -mark-> VERB (get)
        We replaced relcl with acl:relcl, but in the above case, if PART (to) -aux-> VERB (get) -acl:relcl-> X UD requires acl
    """
    if word_node.upos == 'PART':
        word_node.dependency_relation = 'mark'
        gov_node: WordNode
        gov_node = word_node.governor_word
        if gov_node.dependency_relation == 'acl:relcl':
            gov_node.dependency_relation = 'acl'


def oprd_to_ud(word_node: WordNode):
    """ Object Predicate - not sure what it is. So far saw only 2 examples
        I had a dog named Fido
        Entering the room sad is not recommended
    """
    if word_node.upos in ['NOUN', 'PRON', 'PROPN']:
        word_node.dependency_relation = 'obj'
    else:
        word_node.dependency_relation = 'advcl'


"""
Sometimes, Spacy uses 'amod' to modify verbs where it should be 'xcomp'. In UD, 'amod' modifies nominals
'Sue looks great'
    Spacy: (PROPN Sue) -nsubj-> (VERB looks) <-amod- WN (ADJ great)
    UD: (PROPN Sue) -nsubj-> (VERB looks) <-xcomp- WN (ADJ great)
"""


def amod_to_ud(word_node: WordNode):
    gov_node: WordNode
    gov_node = word_node.governor_word
    if gov_node.upos == 'VERB':
        word_node.dependency_relation = 'xcomp'


"""
Spacy uses 'nmod' in some numeric expressions where UD uses 'nummod', for example:
'Sam spent 40 dollars'
    Spacy: (SAM) -nsubj-> (VERB spent) <-dobj- (NOUN dollars) <-nummod- (NUM 40)
    UD: (SAM) -nsubj-> (VERB spent) <-obj- (NOUN dollars) <-nummod- (NUM 40)
This is UD correct. On the other hand
'Sam spent $40'
    Spacy: (SAM) -nsubj-> (VERB spent) <-dobj- (NUM 40) <-nmod- WN (SYM $)
    UD: (SAM) -nsubj-> (VERB spent) <-obj- WN (SYM $) <-nummod- (NUM 40)
Spacy's output in inconsistent and needs to be corrected
"""


def nmod_to_ud(word_node: WordNode):
    gov_node: WordNode
    gov_node = word_node.governor_word
    if word_node.lemma == '$' and gov_node.upos == 'NUM':
        word_node.dependency_relation = gov_node.dependency_relation
        word_node.governor = gov_node.governor
        word_node.governor_word = gov_node.governor_word
        gov_node.dependency_relation = 'nummod'
        gov_node.governor = word_node.index
        gov_node.governor_word = word_node
        redirect_dependants(gov_node, word_node)


"""
Spacy uses 'nummod' where UD specifies 'nmod'
'I live in apartment 71'
    Spacy: (I) -nsubj-> (VERB live) <-prep- (PREP in) <-pobj- (NOUN apartment) <-nummod- WN (NUM 71)
    UD: (I) -nsubj-> (VERB live) <-obl- (NOUN apartment) <-case- (PREP in)
                                                         <-nmod- WN (NUM 71)

We need to check if the governor of 'nummod' is a NOUN immediately preceding the number
"""


def nummod_to_ud(word_node: WordNode):
    gov_node: WordNode
    gov_node = word_node.governor_word
    if gov_node.upos == 'NOUN' and gov_node.index == word_node.index - 1:
        word_node.dependency_relation = 'nmod'


"""
Sometimes Spacy uses advcl which is a predicate modifier instead of acl which is a noun modifier 
'The issues as he sees them'
    Spacy: (NOUN issues) <-advcl- WN (VERB sees)
    this is correct in the case when NOUN is the head of a copula (e.g., 'these are the issues as he sees them')
            otherwise, we need to change it to acl
    UD: (NOUN issues) <-acl- WN (VERB sees)
Copula
'For us to not attempt to solve the problem is for us to acknowledge defeat'
    Spacy: (VERB attempt) -csubj-> BE (is) <-advcl- WN (VERB acknowledge)
    UD: (VERB attempt) -csubj-> WN (VERB acknowledge) <-cop- BE (is)
Except when WN has no subject as in:
'Let's face it, we are tired'
    Spacy: WN (VERB let) -advcl-> BE (are) <-acomp- (ADJ tired)
    UD: (VERB let) <-prataxis- (ADJ tired)

    There are two issues here:
    (1) should (VERB let) or (ADJ tired) be the root?
        if we had 'We are tired, let's face it', (ADJ tired) would be the root
        it really should not matter which one is the root
    (2) how do we get 'prataxis' from 'advcl'?
        maybe we shouldn't
"""


def advcl_to_ud(word_node: WordNode):
    gov_node: WordNode
    gov_node = word_node.governor_word
    if gov_node.lemma == 'be' and find_subj(word_node):  # subj may be too weak a filter
        make_copula(word_node, gov_node)
        # now word_node is the head of the copula
        csubj_node: WordNode
        csubj_node = find_governed(word_node, 'csubj')
        if csubj_node and csubj_node.upos == 'VERB':
            csubj_node.dependency_relation = 'csubj:outer'
    elif gov_node.upos in ['NOUN', 'PRON', 'PROPN'] and not find_governed(gov_node, 'cop'):
        word_node.dependency_relation = 'acl'


"""
ccomp and xcomp are clausal complements. The difference is that the subject of xcomp is 'controlled' - must be the same
as the higher subject or object. We leave it to Spacy to get these right (or wrong)
Since Spacy does not mark copulas and uses the verb to be instead, we need to consider the cases when ccomp and xcomp
depend on 'be'.
There are several cases to consider:
1) 'The problem is Sue left the room'
    Spacy: X (problem) -nsubj-> BE <-ccomp- VERB (left) <-nsubj- Y (Sue)
    first, we make BE into a copula whose head is VERB (left) representing 'Sue left the room', obtaining
    Copula: X (problem) -nsubj-> VERB (left) <-nsubj- Y (Sue)
    now 'left' has two subjects (nsubj) which is not good, but this is how Stanza parses it.
    The proper UD solution is to make the first one nsubj:outer
    UD: X (problem) -nsubj:outer-> VERB (left) <-nsubj- Y (Sue)
2) 'The problem is to leave the room' (or 'His dream is to fly airplanes')
    Spacy: X (problem) -nsubj-> BE <-xcomp- VERB (leave)
    Copula: X (problem) -nsubj-> VERB (leave)
    now, 'problem' is the subject (nsubj) of 'leave' which is not good
    The proper UD solution is again nsubj:outer
    UD: X (problem) -nsubj:outer-> VERB (leave)
    Note, that Stanza does not create a copula in this case, but UD requires it
3) 'It is important that you leave the room'
    Spacy: PRON (it) -nsubj-> be (is) <-acomp- ADJ (important)
                                      <-ccomp- VERB (leave) <-nsubj- PRON (you)
    first, acomp triggers copula:
    Copula: PRON (it) -nsubj-> ADJ (important) <-ccomp- VERB (leave) <-nsubj- PRON (you)
    We could leave it at that, but 'it' here stands for the whole clausal complement 'you leave the room'
    The proper UD solution is to make the clause into a clausal subject:
    UD: PRON (it) -expl-> ADJ (important) <-csubj- VERB (leave) <-nsubj- PRON (you)
    This is similar to the treatment of 'that you leave the room is important'
4) 'It is you who left the room'
    Spacy: PRON (it) -nsubj-> be (is) <-attr- PRON (you)
                                      <-ccomp- VERB (left) <-nsubj- PRON (who)
    Copula: PRON (it) -nsubj-> PRON (you) <-ccomp- VERB (left) <-nsubj- PRON (who)
    One can argue that 'it' stands for 'who left the room' and that ccomp should be made into csubj
    Stanza, on the other hand,  parses this as: PRON (you) <-acl:relcl- VERB (left)
    I would treat it as csubj similar to (3)
5) 'It is important to leave the room'
    Spacy: PRON (it) -nsubj-> be (is) <-acomp- ADJ (important)
                                      <-xcomp- VERB (leave)
    This is similar to (3)
    UD: PRON (it) -expl-> ADJ (important) <-csubj- VERB (leave) <-nsubj- PRON (you)
    similar to: 'to leave the room is important'
6) 'He claimed to be a wizard'
    Spacy: PRON (he) -nsubj-> VERB (claimed) <-xcomp- be <-attr- NOUN (wizard)
    <-attr- triggers copula
    Copula: PRON (he) -nsubj-> VERB (claimed) <-xcomp- NOUN (wizard) <-cop- be
    This is a proper UD

Prataxis
Spacy uses 'parataxes' while UD uses 'prataxis'
It may be hard to distinguish between ccomp and prataxis as in:
'Let's face it, we are annoyed'
    Spacy: (VERB let) -advacl-> BE (AUX are) <-acomp- (ADJ annoyed)
    UD: (VERB let) <-prataxis- (ADJ annoyed)
"""


def comp_to_ud(word_node: WordNode):
    """ 'It is very important that your students respect you' """
    gov_node: WordNode
    gov_node = word_node.governor_word
    subj_node: WordNode
    subj_node = find_subj(gov_node)  # either nsubj or csubj
    if gov_node.lemma == 'be':
        make_copula(word_node, gov_node)  # now word_node is the head of the copula
        if subj_node:  # subj_node has been redirected to word_node by make_copula
            # we need to make subject dependency 'outer' to avoid double subject
            subj_node.dependency_relation += ":outer"
    elif find_governed(gov_node, 'cop'):  # ccomp/xcomp is modifying a copula
        # if copula's subject was 'it' we change ccomp/xcomp into csubj
        if subj_node and subj_node.lemma == 'it':
            subj_node.dependency_relation = 'expl'
            word_node.dependency_relation = 'csubj'


"""
UD does not use pobj (prepositional object).
A typical construction produced by Spacy is:
'We stayed in the room' or 'the chair near the table'
Spacy: X (stayed or chair) <-prep- PREP (in or near) <-pobj- WN (room or chair)
If X is a nominal, UD uses nmod:
    X (chair) <-nmod- WN (table) <-case- PREP (near)
If X is a verb (not 'to be'), UD uses obl:
    X (stayed) <-obl- WN (room) <-case- PREP (in)

We can also have a chain of prepositions:
'we exercise except after dinner'
    Spacy: (exercise) <-prep- SCONJ (except) <-prep- ADP (after) <-pobj- WN (dinner)
        Stanza thinks that 'except' is an ADP, not SCONJ
    UD: (exercise) <-obl- WN (dinner) <-mark- SCONJ (except)  (it would be case if it were ADP)
                                      <-case- ADP (after)
    We make all prepositions point to WN and WN to the governor of the first preposition

Passive
'He was killed by the police'
    Spacy: VERB (kill) <-agent- PREP (by) <-pobj- WN (police)
    UD: VERB (kill) <-obl:agent- WN (police) <-case- PREP (by)

Copula
'We are in the barn' or 'We are out of the woods'
    Spacy: SUBJ (we) -nsubj-> BE (are) <-prep- PREP (in) <-pobj- WN (barn)
    UD: SUBJ (we) -nsubj-> WN (barn) <-case- PREP (in) and WN <-cop- BE (are)

"""


def pobj_to_ud(word_node: WordNode):
    prep_node: WordNode
    gov_node: WordNode
    gov_node, prep_nodes = prep_chain(word_node)
    if not prep_nodes:
        print(f"pobj without preps, word: {word_node.text}")
        return
    if not gov_node.lemma == 'be' or not make_copula(word_node, gov_node):
        prep_node = prep_nodes[0]
        if prep_node.dependency_relation == 'prep':
            if gov_node.upos in ['NOUN', 'PRON', 'PROPN']:
                word_node.dependency_relation = 'nmod'
            else:
                word_node.dependency_relation = 'obl'
        elif prep_node.dependency_relation == 'iobj':  # "give the toys to Mary" give <-dative- with dative transformed into iobj
            word_node.dependency_relation = 'obl'
        elif prep_node.dependency_relation == 'agent':
            word_node.dependency_relation = 'obl:agent'
        else:
            print(
                f"Unknown dep from PREP in: {gov_node.text} <-{prep_node.dependency_relation}- {prep_node.text} <-pobj- {word_node.text}")
            return
        word_node.governor = gov_node.index
        word_node.governor_word = gov_node
    for prep_node in prep_nodes:
        prep_node.governor = word_node.index
        prep_node.governor_word = word_node
        prep_node.dependency_relation = 'case'
        adv_node: WordNode
        adv_node = find_governed(prep_node, 'advmod')
        if adv_node:  # especially on Mondays: ADV (especially) -advmod-> PREP (on)
            adv_node.governor = word_node.index
            adv_node.governor_word = word_node


def prep_chain(word_node: WordNode):
    x: WordNode
    prep_nodes = []
    head = None
    x = word_node.governor_word
    if x.dependency_relation in ['iobj', 'agent']:
        return x.governor_word, [x]
    while not head:
        if x.dependency_relation == 'prep':
            prep_nodes.append(x)
            x = x.governor_word
        else:
            head = x
    return head, prep_nodes


"""
Spacy does not use 'fixed' and does not recognize many expressions such as 'as well as'
But it often uses -pcomp-> (complement preposition) to indicate a multi-word SCONJ such as 'because of' in which case
we replace -pcomp-> with -fixed->
'He cried because of you'
    Spacy:  X (cried) <-prep- SCONJ (because) <-pcomp- ADP (of)
                                              <-pobj- Y (you)
    UD:     SCONJ (because) <-fixed- ADP (of)

UD does not use pcomp (prepositional complement), considers them non-core and uses advcl with some exceptions:
When pcomp complements a copula, it becomes its head
'We are almost near there'
    Spacy:  X (we) -nsubj-> be <-prep- PREP (near)  <-advmod- ADV (almost)
                                                    <-pcomp- WN (there)
    UD:     X (we) -nsubj-> WN (there) <-case- PREP (near)
                                       <-advmod- ADV (almost)
                                       <-cop- AUX (be)

'I am tired of waiting'
    Spacy:  X (tired) <-prep- ADP (of) <-pcomp WN (waiting)
    UD:     X (tired) <-advcl- WN (waiting) <-case- PREP (of)
'They heard about you missing classes'
    Spacy:  X (heard) <-prep- SCONJ (about) <-pcomp- WN (missing)
    UD:     X (heard)  <-advcl- WN (missing) <-mark- SCONJ (about)
"""


def pcomp_to_ud(word_node: WordNode):
    prep_node: WordNode
    gov_node: WordNode
    prep_node = word_node.governor_word
    if word_node.upos == 'ADP' and prep_node.upos == 'SCONJ':
        word_node.dependency_relation = 'fixed'
        return
    gov_node = prep_node.governor_word
    if prep_node.dependency_relation != 'prep':
        print(
            f"No prep dependency in: {gov_node.text} <-{prep_node.dependency_relation}- {prep_node.text} <-pcomp- {word_node.text}")
        return
    if gov_node.lemma == 'be':
        make_copula(word_node, gov_node)
    else:
        word_node.dependency_relation = 'advcl'
        word_node.governor = gov_node.index
        word_node.governor_word = gov_node
        prep_node.governor = word_node.index
        prep_node.governor_word = word_node
    dep = 'case'
    if prep_node.upos == 'SCONJ':
        dep = 'mark'
    prep_node.dependency_relation = dep
    # there may be some nodes (e.g., adverbs) dependent on prep_node - we need to move them to word_node
    redirect_dependants(prep_node, word_node)


"""
Spacy seems to use attr only as the direct nominal object of the verb to be
Existential - not a copula
'There is a ghost in the room'
    Spacy: there -expl-> BE (is) <-attr- WN (ghost)
    UD: there -expl-> BE (is) <-nsubj- WN (ghost)
Copula
'Mary is a doctor'
    Spacy: X (Mary) -nsubj-> be(AUX is) <-attr= WN (doctor)
    UD: X (Mary) -nsubj-> WN (doctor) <-cop- be(AUX is)
"""


def attr_to_ud(word_node: WordNode):
    be_node: WordNode
    subj_node: WordNode
    be_node = word_node.governor_word
    if be_node.lemma != 'be':
        print(f"Dep attr used with something other than to be: {be_node.text} <-attr- {word_node.text}")
        return
    subj_node = find_governed(be_node, 'expl')
    if subj_node:  # we only need to change the relation
        word_node.dependency_relation = 'nsubj'
        return
    make_copula(word_node, be_node)


"""
UD does not use acomp (adjectival complement)
Spacy uses acomp in two cases:
Copula
'The truck is green'
    Spacy: SUBJ (truck) -nsubj-> be(AUX is) <-acomp- WN(ADJ green)
    UD: SUBJ (truck) -nsubj-> WN(ADJ green) <-cop- be(AUX is)
'The trucks looks green'
    Spacy: SUBJ (truck) -nsubj-> VERB(looks) <-acomp- WN(ADJ green)
    UD: SUBJ (truck) -nsubj-> VERB(looks) <-xcomp- WN(ADJ green)

Sometimes, Spacy treats a passive as an acomp, which is probably a mistake:
'The speech was well received'
    Spacy: SUBJ (speech) -nsubj-> BE (AUX was) <-acomp- VERB (received)
'The speech was well argued'
    Spacy: SUBJ (speech) -nsubjpass-> VERB (received) <-auxpass- BE (AUX was)
"""


def acomp_to_ud(word_node: WordNode):
    acomp_node: WordNode
    acomp_node = word_node.governor_word
    if acomp_node.lemma == 'be':
        if word_node.upos == 'VERB' and word_node.features.get('Aspect') == 'Perf':
            # this is to fix the case when acomp connects a verb instead of an adjective
            make_passive(acomp_node, word_node)
        else:
            make_copula(word_node, acomp_node)
    else:
        word_node.dependency_relation = 'xcomp'


"""
Spacy's advmod (adverbial modifier) should be the same as UD's, but there are some subtle issues
Spacy always parses 'when', 'where' and 'how' as SCONJ but uses advmod to connect them to the verb while
UD considers these ADV unless they are introducing a subordinate clause
'When do you want to talk?'
    Spacy: SCONJ (when) -advmod-> VERB (talk)
    UD: ADV (when) -advmod-> VERB (talk)
    Stanza parses this as: ADV (when) -advmod-> VERB (want), which is not correct at least according to a UD example
'He was upset when we talked' - here 'when' is actually an SCONJ
    Spacy: SCONJ (when) -advmod-> VERB (talked)
    UD: SCONJ (when) -mark-> VERB (talked)

How can we tell if an SCONJ in SCONJ -advmod-> VERB should be an adverb (ADV) and when it is a true SCONJ?
We can look if the verb is the head of a subordinate clause, e.g., VERB -advcl-> X
If it is, we change SCONJ -advmod-> VERB into CONJ -mark-> VERB
otherwise, we change SCONJ into ADV
"""


def advmod_to_ud(word_node: WordNode):
    if word_node.upos == 'SCONJ':
        if word_node.governor_word.dependency_relation == 'advcl':
            word_node.dependency_relation = 'mark'
        else:
            word_node.upos = 'ADV'


"""
In Spacy, conjunction is cc attached to the first conjunct, in UD to the second
'bread and butter'
    Spacy: CONJ (and) -cc->  X (bread) <-conj- WN (butter)
    UD: X (bread) <-conj- WN (butter) <-cc- CONJ (and)
'bread, butter and jam'
    Spacy: Y (bread) <-conj- X (butter) <-conj- WN(jam)
                                        <-cc- CONJ (and)
    UD: Y (bread) <-conj- X (butter)
                  <-conj- WN (jam) <-cc- CONJ (and)

orphan
UD gives an example:
'Mary won gold and Peter bronze'
    UD: (Mary) -nsubj-> (VERB win) <-obj- (NOUN gold)
                                   <-conj- (Peter) <-orphan- (NOUN bronze)
    probably because (Peter) is considered the head of the clause 'Peter [won] bronze'
    Spacy: (Mary) -nsubj-> (VERB win) <-dobj- (NOUN gold)
                                      <-conj- (NOUN bronze) <-nsubj- (Peter)
    probably because it considers (bronze) the head of the clause without the verb
    Spacy's treatment is more consistent with the treatment of copulas:
'Sue is a doctor and Mary is a pilot'
    Spacy: (SUE) -nsubj-> BE (is) <-attr- (doctor) <-conj- (pilot) <-nsubj- (Mary)
    UD: (SUE) -nsubj-> (doctor) <-cop- BE (is)
                                   <-conj- (pilot) <-nsubj- (Mary)
If we want to follow the UD example, we need to check the chain:
VERB <-conj- NOUN1 <-nsubj- NOUN2
and replace it with
VERB <-conj- NOUN2 <-orphan- NOUN1
"""


def conj_to_ud(word_node: WordNode):
    conj_node: WordNode
    gov_node: WordNode
    gov_node = word_node.governor_word
    conj_node = find_governed(gov_node, 'cc')  # we find the first but there might be more
    if conj_node:
        conj_node.governor = word_node.index
        conj_node.governor_word = word_node
    if gov_node.upos == 'VERB' and word_node.upos in ['NOUN', 'PRON', 'PROPN']:
        nsubj_node: WordNode
        nsubj_node = find_governed(word_node, 'nsubj')  # NOUN2 above
        if nsubj_node:
            nsubj_node.dependency_relation = 'conj'
            nsubj_node.governor = gov_node.index
            nsubj_node.governor_word = gov_node
            word_node.dependency_relation = 'orphan'
            word_node.governor = nsubj_node.index
            word_node.governor_word = nsubj_node
            if conj_node:
                conj_node.governor = nsubj_node.index
                conj_node.governor_word = nsubj_node
    if gov_node.dependency_relation == 'conj':
        gov_gov_node: WordNode
        gov_gov_node = gov_node.governor_word
        word_node.governor = gov_gov_node.index
        word_node.governor_word = gov_gov_node


"""
UD does not use npadvmod - noun phrase adverbial modifier
'I ate an apple today'
    Spacy: (ate) <-npadvmod- WN (today)
    UD: (ate) <-obl:tmod- WN (today)
'I am 73 years old'
    Spacy: (PRON I) -nsubj-> BE (am) <-amod- (ADJ old) <-npadvmod WN (years) <-nummod- (NUM 73)
    UD: (PRON I) -nsubj-> (ADJ old) <-cop- BE (am)
                                    <-obl:tmod- WN (years) <-nummod- (NUM 73)
'The stick is 6 feet long'
    Spacy: (NOUN stick) -nsubj-> BE (is) <-amod- (ADJ long) <-npadvmod WN (feet) <-nummod (NUM 6)
    UD: (NOUN stick) -nsubj-> (ADJ long) <-cop- BE (is)
                                         <-obl:npmod- WN (feet) <-nummod- (NUM 6)
    Stanza uses obl:npmod, but this dependency is not listed in UD
'The noose eased a fraction'
    Spacy: (NOUN noose) -nsubj-> (VERB eased) <-npadvmod- WN (fraction)
    UD: (NOUN noose) -nsubj-> (VERB eased) <-obl:npmod- WN (fraction)
    Stanza parses this as: (VERB eased) <-obj- WN (fraction), which is probably a mistake
'IBM earned $5 a share'
    Spacy: (PRPN IBM) -nsubj-> (VERB earned) <-dobj- (NUM 5) <-nmod- (SYM $)
                                                             <-npadvmod- WN (share)
    UD: (PRPN IBM) -nsubj-> (VERB earned) <-obj- (SYM $) <-nummod- (NUM 5)
                                          <-obl:npmod- WN (share)
    Stanza parses this as: (VERB earned) <-obj- WN (share), which is probably a mistake as it creates 2 obj
'The silence is itself significant'
    Spacy: (NOUN silence) -nsubj-> BE (is) <-amod- (ADJ significant) <-npadvmod- WN (PRON itself)
    UD: (NOUN silence) -nsubj-> (ADJ significant) <-cop- BE (is)
                                                  <-obl:npmod- WN (PRON itself)
    Stanza parses this as: (ADJ significant) <-advmod- WN (PRON itself), which is probably a mistake
'I hate it, the most'
    Spacy: (PRON I) -nsubj-> (VERB hate) <-dobj- (PRON it)
                                         <-npadvmod- WN (ADV most)
    UD: (PRON I) -nsubj-> (VERB hate) <-obj- (PRON it)
                                      <-nobl:npmod- WN (ADV most)
    Stanza parses this as: (VERB hate) <-parataxis- WN (ADV most), which is probably a mistake
'Gus, take it easy!'
    Spacy: WN (PROPN Gus) -npadvmod-> (take)
    UD: WN (PROPN Gus) -vocative-> (take)
'Steve Jones sj@abc.xyz University of Arizona'
    Spacy: (PROPN Steve) -compound-> (PROPN Jones) <-npadvmod- WN (PROPN University)
    UD: (PROPN Steve) <-flat- PROPN Jones)
                      <-list- WN (PROPN University)

Since it is very difficult to tell when to translate npadvmod into obl:tmod, obl:npmod. vocative or list,
we can make it into obl:npmod in all cases

We still need to transform a chain of npadvmod's into a parallel obl:npmod dependency:
'Are you free for lunch some day this week?'
    Spacy: (ADJ free) <-npadvmod- (day) <-npadvmod- (week)
    UD: (ADJ free) <-obl:tmod- (day)
                   <-obl:tmod- (week)

"""


def npadvmod_to_ud(word_node: WordNode):
    gov_node: WordNode
    gov_node = word_node.governor_word
    if gov_node.dependency_relation == 'obl:npmod':
        gov_node = gov_node.governor_word
    word_node.dependency_relation = 'obl:npmod'
    word_node.governor = gov_node.index
    word_node.governor_word = gov_node


""" compound vs flat

Both Spacy and UD use the 'compound' dependency, but they do it somewhat differently.
UD uses 'flat' with proper names
'Hilary Rodham Clinton'
    Spacy: (Hilary) -compound-> (Clinton) <-compound (Rodham)
    UD: (Hilary) <-flat- (Rodham)
                 <-flat- (Hilary)
'New York'
    Spacy: (New) -compound-> (York)
    UD: (New) <-flat- (York)
'Mr. Smith'
    Spacy: (Mr.) -compound-> (Smith)
    UD: (Mr.) <-flat- (Smith)
'French actor Gaspar Ulliel'
    Spacy: (ADJ french) -amod-> (actor) -compound-> (Ulliel) <-compound- (Ulliel)
    UD: (ADJ french) -amod-> (actor) <-flat- (Gaspar)
                                     <-flat- (Ulliel)
On the other hand, UD also uses 'compound' in:
'Natural Resources Conservation Service'
    Spacy: (ADJ Natural) -amod-> (PROPN resources) -compound-> (PROPN Service) <-compound- (PROPN conservation)
    UD: (ADJ Natural) -amod-> (PROPN resources) -compound-> (PROPN Service) <-compound- (PROPN conservation)

Since it is impossible to tell the difference in Spacy output between 'Gaspar Ulliel' and 'Conservation Service'
We have no basis for transforming 'compound' into 'flat'

UD uses 'flat' in dates and numbers such as '1 December 2022' and 'three hundred twenty one' while Spacy uses 'nummod'
and 'compound'. We leave Spacy's output since these expressions should be identified by the NER component

appos vs list

In enumerations such as 'We saw long lines, silly rules, rude staff'
Spacy uses 'appos' to link (rules) and (staff) to (lines)
UD uses 'list' for the same
I don't see any point in changing 'appos' into 'list'
"""


def dep_to_ud(word_node: WordNode):
    """ this may be a hack, we need to see more cases of dep: dep"""
    gov_node: WordNode
    gov_node = word_node.governor_word
    if gov_node.upos == 'VERB':
        word_node.dependency_relation = 'ccomp'


"""
We don't create copulas from adverbs during the first pass because there may be several adverbs
Spacy attaches all of them to the BE node via advmod
If there are several such adverbs, we want the last to be the head of the copula
'I am here'
    Spacy: X (I) -nsubj-> BE (am) <- advmod- WN (ADV here)
    UD: X (I) -nsubj-> WN (ADV here) <-cop- BE (am)
'We are in here'
    Spacy: X (we) -nsubj-> BE (are) <-advmod- ADV (here) <-advmod- WN (ADV in)
    UD: X (we) -nsubj-> ADV (here) <-case- WN(PREP in)
                                   <-cop- BE (are)
"""


def fix_advmod_cop(sent: SentenceNode):
    word_node: WordNode
    for word_node in reversed(sent.word_nodes):
        if word_node.dependency_relation == 'advmod':
            gov_node: WordNode
            gov_node = word_node.governor_word
            if gov_node.lemma == 'be' and gov_node.dependency_relation != 'cop':
                make_copula(word_node, gov_node)


def make_copula(head_node: WordNode, be_node: WordNode):
    # UD does not convert existential predications into copulas
    expl_node: WordNode
    expl_node = find_governed(be_node, 'expl')
    if expl_node and expl_node.lemma == "there":
        return False
    # make head_node the copula predicate and move governor dependency from be_node to head_node
    head_node.governor = be_node.governor
    head_node.governor_word = be_node.governor_word
    head_node.dependency_relation = be_node.dependency_relation
    # make be_node dependent (cop) on head_node
    be_node.governor = head_node.index
    be_node.governor_word = head_node
    be_node.dependency_relation = 'cop'
    be_node.upos = 'AUX'
    redirect_dependants(be_node, head_node)  # if X is dependent on be_node then make it dependent on head_node
    return True


def redirect_dependants(from_node: WordNode, to_node: WordNode):
    sent: SentenceNode
    sent = from_node.sentence_node
    word_node: WordNode
    for word_node in sent.word_nodes:
        if word_node == from_node or word_node == to_node:
            continue
        if word_node.governor_word == from_node:
            if word_node.dependency_relation in ['prep', 'attr', 'advcl', 'advmod', 'acomp', 'xcomp', 'dep', 'acl',
                                                 'nsubj', 'obj', 'csubj', 'ccomp', 'mark', 'cop', 'npadvmod', 'conj',
                                                 'prataxis', 'punct', 'cc', 'nsubj:outer', 'csubj:outer']:
                word_node.governor = to_node.index
                word_node.governor_word = to_node


def make_passive(be_node: WordNode, verb_node: WordNode):
    subj_node: WordNode
    subj_node = find_governed(be_node, 'nsubj')
    rel = 'nsubj:pass'
    if not subj_node:
        subj_node = find_governed(be_node, 'csubj')
        rel = 'csubj:pass'
    if subj_node:
        subj_node.governor = verb_node.index
        subj_node.governor_word = verb_node
        subj_node.dependency_relation = rel
    verb_node.governor = be_node.governor
    verb_node.governor_word = be_node.governor_word
    verb_node.dependency_relation = be_node.dependency_relation
    be_node.governor = verb_node.index
    be_node.governor_word = verb_node
    be_node.dependency_relation = 'aux:pass'


def find_governed(word_node: WordNode, dep):
    sent = word_node.sentence_node
    wn: WordNode
    for wn in sent.word_nodes:
        if wn.governor_word == word_node and wn.dependency_relation == dep:
            return wn
    return None


def find_subj(word_node: WordNode):
    sent = word_node.sentence_node
    wn: WordNode
    for wn in sent.word_nodes:
        if wn.governor_word == word_node and\
                wn.dependency_relation in ['nsubj', 'csubj', 'nsubj:pass', 'csubj:pass', 'nsubj:outer', 'csubj:outer']:
            return wn
    return None


def add_entities(sent: SentenceNode, spacy_sent: SpacyDoc):
    # add extracted NERs
    spacy_span: spacy.tokens.Span
    for spacy_span in spacy_sent.ents:
        span_word_nodes = find_span_words(sent, spacy_span)
        phrase_head_word = None
        for word_node in span_word_nodes:
            if word_node.governor_word is None or word_node.governor_word not in span_word_nodes:
                phrase_head_word = word_node
                word_node.ner = {'text': spacy_span.text, 'words': span_word_nodes,
                                 'span': [spacy_span.start_char, spacy_span.end_char], 'type': spacy_span.label_}
                break
        for word_node in span_word_nodes:
            if word_node != phrase_head_word:
                word_node.ner_head_word = phrase_head_word


def find_span_words(sent: SentenceNode, spacy_span: spacy.tokens.Span):
    """ returns word nodes that are within Spacy span"""
    span_words = []
    for word_node in sent.word_nodes:
        if word_node.span[0] >= spacy_span.start_char and word_node.span[1] <= spacy_span.end_char:
            span_words.append(word_node)
    return span_words

