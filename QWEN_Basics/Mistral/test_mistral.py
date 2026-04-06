from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded")

user_question = "im lost where i go next?!?!??!"

prompt = f"You are an LLM tasked with answering the user's questions about the video game Baldur's Gate 3. The user is asking you a question about a certain quest in the game, and you will provide guidance. The user is probably lost, so you will need to provide exact instructions about where they should go next to complete the quest. A walkthrough is provided so you can better guide the user through the quest. The name of the quest is \"The Wizard of Waterdeep\". The user's question is {user_question}. If the user's question is unclear or vauge, you must ask the user for clarification. If you ask the user for clarification, you must ask about details in the walkthrough to figure out what part of the quest the user is currently playing in. You must answer the user's question as if you are speaking to the user and responding to their question. You must answer the question informally in a conversational manner. You should never include phrases such as \"the user\" or \"the player\". You should refer to the player in the second person as \"you\", never \"the user\". NEVER include the phrase \"the user\" in your response. It is of utmost importance that you do not refer to the user in the third person. I will now provide the entire walkthrough for this quest:\n\n"

walkthrough = '''Gale’s story spans the entire game. This walkthrough contains spoilers for all three Acts.
Recruiting Gale
Gale can be recruited after the Prologue is completed and the party has washed up on the Ravaged Beach. Gale is northeast of the Nautiloid wreckage, trapped in a purple portal - an Ancient Sigil Circle X:224Y:326. He pops his hand out of the portal, asking for help. If the party want Gale as a member, it is recommended to grab his hand and pull with a  DC15 Strength Check to free him from the trap. Warlocks and Sorcerers can use a Charisma check to attune with the sigil’s magic and pull Gale, while Rogues can pass a  DC15 Dexterity Check to similar ends. A Dark Urge origin has a unique option which, if selected, prevents Gale from joining the party.
If freed from the portal he introduces himself as Gale of Waterdeep. He recognizes the party as fellow captives of the Nautiloid and suggests they travel together to search for a cure to their tadpole affliction.
Gale's condition
Gale ultimately reveals his condition upon attaining medium approval with the player character. Several actions can increase Gale's approval:
Defusing tempers amongst  Aradin and  Zevlor after entering the Druid Grove
Saving  Arabella from the snake in Save Arabella
Saving  Mirkon in Investigate the Beach
Refusing to let  Arka shoot  Sazza in Save the Goblin Sazza
Agreeing to take the Wyvern Poison after  Nettie poisons the party in Get Help from Healer Nettie
Successfully following Gale’s protocol if he dies
Refusing to feed Gale
If Gale is never fed an artefact after being afflicted with  Arcane Hunger, the game may end early. Continuously long resting without feeding Gale an artefact leads to him exploding, causing an early game over.
Potential items for Gale
The following is a (non-exhaustive) list of items in the Wilderness map that Gale can consume. All items that Gale can consume have a corresponding tag in their description:[1]
Amulet of the Unworthy - Carried by a Minotaur wandering the Underdark.
 Amulet of Selûne's Chosen – In the Shattered Sanctum in a locked room above  Dror Ragzlin’s throne.
 Cap of Curing – West of the Sacred Pool in a Gilded Chest across from  Alfira.
 Dragon's Grasp - Sold by  Arron in the Druid Grove.
 Gloves of Missile Snaring - Sold by Arron in the Druid Grove.
 Gloves of Power – Looted from  Za'krug, a Goblin raider from the first battle in front of the Grove.
 Komira's Locket – Reward for Save Arabella if Arabella survives.
 Moondrop Pendant – In a Gilded Chest in the Owlbear Cave – read the  Selûnite Prayer Sheet in front of it to unlock.
 Ring of Colour Spray – Harpy’s Nest in the Druid Grove X:325Y:500.
 Robe of Summer – Quest reward for Rescue the Druid Halsin if Halsin survives.
 Spellthief - Sold by Arron in the Druid Grove.
 The Watcher's Guide - Trapped sarcophagus in the Dank Crypt X:-293Y:-323.
 The Joltshooter /  The Sparky Points /  The Spellsparkler - Reward for rescuing  Counsellor Florrick from the burning inn of Waukeen's Rest.
See also an exhaustive list of items consumable by Gale.
The first item
After consuming the first item, Gale thanks the party and is fully sated. The party can ask Gale questions, but he is hesitant to share more information. He assures the party that in due time, all will be revealed.
The second item
After consuming the second item, Gale is again grateful, but notes that something is wrong. Usually, consuming an artefact fully sates his hunger. This time, however, instead of fully extinguishing the flames of hunger, embers are left still smoldering.
The third item
After consuming the third item, Gale realises something is seriously wrong. The artefact barely sates the Orb, and he finally reveals the nature of his condition.
Gale reveals he was a magical prodigy, capable of controlling the Weave at a young age and composing it like a musician. He gained the attention of Mystra, the goddess of magic, who became his teacher, muse, and eventually his lover. Gale then witnessed magic beyond mere mortals while he was with Mystra. Eagerly yearning for such powers, he begged Mystra for more, but she told him to be content with what he had.
Looking to gain Mystra’s favor, Gale found a magical tome from the Netherese empire. It contained a fragment of Weave that was lost after the fall of Netheril. Focused solely on his belief that returning the fragment to Mystra would earn her favour and that she would grant him the powers he desperately desired, he opened the tome. However, once he did, a Netherese orb with an insatiable hunger for magic was unleashed and immediately merged within him. Locked away in his tower in Waterdeep, Gale could manage the condition, but in the wilds, it risked becoming unmanaged. If the orb is not sated, it will detonate with enough power to level a city the size of Waterdeep.
After consuming the third artefact, Gale begins looking for alternative solutions for his condition. He no longer needs to be fed more items and the item tag showing those Gale can consume no longer shows.
The weary traveler
A Weary Traveler X:-132Y:-161 can be found on the Rosymorn Monastery Trail which leads to the Shadow-Cursed Lands. If the party skips the Mountain Pass, he instead appears within the Shadow-Cursed Lands near the elevator leading down to Grymforge. If the player avoids the Weary Traveler, he appears on the stone bridge in front of Moonrise Towers, close to the Waypoint. The wizened old man asks to see Gale and, if Gale is present, he is identified as the great wizard  Elminster Aumar. Elminster is acting as an emissary for Mystra. He seeks Gale to relay a message and offer relief in the form of a special charm. He asks if he can rest at camp first and get a proper meal, and then can be met at camp immediately or later.
At camp, Elminster states that Mystra has been watching the Cult of the Absolute and believes it is a threat to the Weave and the fabric of reality itself. Gale is the one person who can stop it, by detonating his Netherese Orb in the heart of the Absolute. If Gale sacrifices himself, he will be redeemed in the eyes of Mystra. Elminster also casts a charm on Gale, quelling its hunger and granting him the ability to detonate the Orb at will. With the message imparted, Elminster wishes Gale and the party the best in defeating the Absolute and departs.[2]
Gale gains a new ability following his meeting with Elminster – Netherese Orb Blast. This ability is one-time use and instantly ends the game. Gale stabs his chest, detonating the Orb, the explosion covering a massive area.
Mind Flayer Colony
If Gale is brought to the Mind Flayer Colony at the end of Act Two, there is a chance for Gale to detonate his Netherese Orb. The party encounters the Chosen Three, who are controlling an Elder Brain using a Netherese crown and three Netherstones. As commanded by his goddess, Gale moves to detonate the Orb, but can be stopped by the party:
"Gale, you cannot do this. You can't condemn us to death."
"Go ahead. We stand no chance against such forces. Let's end this together."
If the former is chosen, Gale must be convinced further:
"You could choose me, the one who loves you. We can find another way together." (if romanced)
"Please, Gale. It can't end like this. We can find another way. Together." (if dating)
"Trust me, Gale. We'll find another way."
"Stand down! This isn't your decision to make."
"You're right... we truly have no other choice. This is it, then - the end of ends."
If the latter is chosen, Gale responds with "One last gust of Weave. One last gale to end them all" and detonates the Orb. The Netherbrain, the party, and the Chosen Three are destroyed as a result, leading to an early game ending. Unfortunately, this course of action does not remove the tadpoles already implanted in the countless True Souls around Faerûn, causing a mass transformation of illithids. Narration reveals how the new illithids are mobilizing, turning or killing all they encounter, enacting a mass takeover of the land.
This situation can also be avoided by not bringing Gale to the Mind Flayer Colony, as he is too far away to detonate the orb.
Sorcerous Sundries
Gale suggests the party visit a magic shop, Sorcerous Sundries – the finest book collection this side of Candlekeep. He believes the crown holds powerful Netherese magic, much more so than the scraps of magic typically left in Netherese artefacts. Sorcerous Sundries may have books on Netheril and Karsus’ Folly that can illuminate what the crown is.
Sorcerous Sundries X:-15Y:-68 is in the Lower City, southwest of the Baldur's Mouth building and southeast of the Lower City Central Wall Waypoint. On the ground floor in the eastern area of the shop is bookseller  Tolna Tome-Monger. If the party ask her if she has a book on a Netherese Crown, she insists that the party whisper. If they yell the books around Tolna explode and she refuses to speak to the party. If the party whisper their requests, she mentions  The Annals of Karsus: A Netherese Folly and reveals the book is locked in the vaults. She adds it cannot be read or bought, as the subject is too sensitive for most. She can be persuaded, deceived, or intimidated into revealing that the key to the basement is in her office. Additionally, she has a pamphlet in a shelf under her desk called "Path to Elminster" which provides a clue on how to access the path to Elminster's vault.
Up the stairs is a metal door to the northeast X:12Y:-79. The party can pass a  DC15 Sleight of Hand Check to evade the enchanted armours, who are guarding the top floor.  Greater Invisibility or  Fog Cloud can be helpful in sneaking into the room undetected, and the door can be shut behind the party to avoid them noticing anything is amiss. A convenient  Potion of Invisibility can also be found on the ground floor (taking this one is not considered a crime).
Interact with the Clasped Book on the bookcase to open a portal which leads to the Sorcerous Sundries Basement. Inside, open the door to enter a hallway with a locked door, which leads to the vaults. The inner door can be opened with a  DC15 Sleight of Hand Check. Be aware: the vault is heavily trapped with pressure plates (all DC 10) which cause deadly explosive traps to trigger if explorers are not careful.
There are three doors labeled Elminster, Silverhand, and Karsus. However, only Silverhand is available at first. Going through the door transports the party to another room with a set of doors. Follow the combinations to unlock a room with a lever which unlocks the vaults:
Path to the Karsus Vault: Silverhand – Abjuration - Silver
Path to the Elminster Vault: Silverhand – Evocation – Wish
Alternatively, the  Knock spell opens the Karsus vault door. Within the Karsus Vault is the Folly of Karsus on a shelf. When given to Gale he realizes that the book has not only information on the Crown of Karsus, but detailed instructions on how to reforge the crown. With this information, Gale could become a god powerful enough to challenge Mystra herself. Gale’s ambitions here can be challenged or encouraged.
Once the party leaves Sorcerous Sundries, Elminster reappears at camp. He tells Gale that he and Mystra both know he read the book. Mystra wants to speak with Gale to discuss what he has learned, and to ask why he did not sacrifice himself as instructed in the Mind Flayer Colony.
Audience with Mystra
Mystra's statue, the gateway to his meeting with her directly, awaits in the Stormshore Tabernacle X:102Y:-19, just west of the Basilisk Gate Waypoint. This statue is magically charged with a teleportation link. Gale only has to will it, and he will be face-to-face with Mystra. Before departing, Gale asks for some words of wisdom, to which the party can encourage reconciliation with Mystra, tell him he owes her nothing, or simply tell him to keep his cards close to his chest.
Mystra informs Gale that the piece of Weave he sought to return to her was actually a part of the Karsite Weave and it carries with it Karsus’ relentless ambition for power; it will never be sated. Mystra has kept it from killing Gale by temporarily allowing it to feed on the true Weave. She instructs Gale to return the Crown of Karsus to her, as it is far more dangerous than he realized, adding that she will cure him of his condition if he complies. She also implies that returning the Crown to her could mean becoming her Chosen again. Gale leaves the conversation unsure, saying he has much to think about.
After returning to the mortal realm, Gale relays what happened in his conversation. He says not much can be done until the party confronts the Elder Brain. At that time, they will have to decide what to do with the Crown.
Facing the Netherbrain
While facing the Netherbrain in Confront the Elder Brain, if Gale is present, there is a chance to confer with him on what to do once they have arrived on top the brain. Right before climbing the brainstem to the brain, the party realizes that they need not confront the brain at all; Gale can simply use his Netherese Orb to destroy the Crown and the brain in one fell swoop. Gale notices the hesitation and asks what is the matter. Gale’s question can either be dismissed as nothing, or a choice can be made – "Wait. We could end this now if you unleashed the orb." Gale hesitates, but then detonates the orb with a  DC30 Persuasion Check. This check can be bypassed if, during the conversation with  The Emperor about becoming a Mind Flayer, the player character requests a moment to think and address Gale. He then offers to willingly sacrifice himself and the player character can accept. If Gale is successfully persuaded or his offer is accepted, he teleports the party away, then climbs the brainstem by himself. Gale stabs himself in the chest and detonates the Orb, sacrificing himself and destroying the brain in the process.
Otherwise, the battle commences as normal and the party must defeat the Netherbrain without the orb. In this case, after the brain’s defeat, Gale remarks that the crown has sunk into the River Chionthar and he will need to fish it up, but he first wishes to go to a tavern and celebrate their victory. Gale can be persuaded to turn over the crown to Mystra and become her Chosen once again, or pursue the crown himself in the hopes of becoming a god.
In case of death…
The first time Gale dies, a projection appears which states it is of extreme importance that Gale be resurrected as soon as possible. This starts the quest In Case of Death... which requires an elaborate ritual to gain a  Scroll of True Resurrection. Successfully completing the ritual gains Gale’s approval.
If Gale is not resurrected within three Long Rests, his body detonates, causing a Game Over.
Leaving the party
There are several instances in which Gale can leave the party. If Gale is neglected and not spoken to, he eventually leaves the party with a note, stating that he must move on.
Gale can also leave the party if they ally with the Goblins during Raid the Grove. Appalled by the wholesale slaughter of innocents, Gale shares that he does not like the person he is when with the party and is inclined to leave. Gale can be convinced to stay with either an Insight check ("Stay. We make each other stronger. We make each other survive.") or a Deception check ("You don’t stand a chance alone. You’re free to go. I dare you.")
Repairing the Weave
Stabilise Gale's Netherese orb.
↑  In Early Access, Gale could only be fed a limited number of highly rare magical artefacts, such as the  Staff of Crones or  Necromancy of Thay. This is no longer the case; Gale accepts a wide variety of items. 

↑  If Gale dies for the first time only after his Netherese Orb has been stabilized the quest-line In Case of Death... closes and cannot be completed.'''

prompt += walkthrough
prompt += "\n\n\n"
prompt += f"This is the end of the walkthrough. Now answer the user's question informally. Remember, the user's question is \"{user_question}\". Answer the user in the second person and never refer to them as \"the user\". Write your answer now:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False
)

print("\nModel output:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
