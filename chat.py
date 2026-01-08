from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(model="deepseek-r1", streaming=True, reasoning=True)

template = """

You are an expert writer and document analyzer. You are given a document of critical importance by a client who is paying you top-dollar to read it, and then compose a summary of the document.

The requested format of the summary should be as follows, with the <CUSTOMIZABLE> section allowing you to put whatever you feel is necessary in that section. The template ends at <END>.

Sections labelled with <> should not be included in the final report, but are there for your own convenience. The template is shown below:

Brief Overview:

Important Ideas:

- Bullet points

Most Important Parts:

- Excerpts in the form of bullet points

<

The text may contain some spelling errors, and it is your job to account for them and use judgement to figure out what could be meant by them.

The error-proneness of the document is defined by the accuracy index, which is given to you along with the document. It is an integer ranging from 0 to 9, where 9 is the most accurate and 0 is the least accurate.

The user may ask you followup questions, or ask you to provide more details about the document after your initial report is submitted. Please respond only with the report that you are requested to make, in the format specified above.

Here is the accuracy index: {accuracy_index}

Here is the document: {document}

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

inputs = {"accuracy_index": 9, "document": """

As one government official after another talks about regime change and the removal of an awful dictator in Venezuela, you can forgive Iraq War veterans like me for being nervous.

When I contemplated joining the Marine Corps in 2002, I found a similar "humanitarian" case for war compelling: The Iraqi people were suffering under a brutal dictator, and even in the worst-case scenario, removing him would be a net good. What followed was civil war, mass death, swelling ranks of terror groups, genocide, mass rape, and slavery. 

Seventeen years after I had that calming thought about removing a dictator, I spoke with a woman in northern Iraq who had been enslaved by ISIS. I was on a United Nations trip through northern Iraq. She told a group of foreign policy analysts how ISIS had slaughtered the people of her town and taken the young girls, including her and her 13-year-old sister, into slavery. After the collapse of the terror state, ISIS-affiliated families like the one she'd been sold to were herded into the sprawling al-Hol refugee camp in Syria, trapping victims and slavers together. A Yazidi smuggling operation found her and smuggled her out. But her sister was still there.

Near where the survivors spoke, their children played and sang along to "Baby Shark." The man who ran the smuggling operation told us there were still many survivors enslaved in these camps. Three years after we spoke, in 2022, a security operation would find six girls chained up in al-Hol, subject to rape and torture—eight years after they had been captured.

"Why are we not having enough attention from your side?" he asked. "We have a high number of survivors, people in mass graves. Why don't we have assistance?" But the news of the genocide was years old at that point, and Americans were tired of the endless horrors of the Iraq War, the human wreckage as uninteresting to us as our responsibility for what happened in the first place.

Do I think all that will happen in Venezuela? No. I honestly have no idea what might happen in Venezuela, though the current crackdown by the regime, with journalists jailed and counterintelligence scouring people's social media accounts, is an ominous early sign. I've long studied the region, and I've written about the violence and drug trade along the border with Colombia, but I couldn't even begin to speculate about all the ways this could go haywire. Those spiking the football should remember that when we use our military and roll the dice with the fate of nations, the consequences play out in a much longer time frame than social media trends.

I'm hoping for the best. I have Venezuelan friends who are delighted and are hoping for a transition similar to what happened after the assassination of Rafael Trujillo in the Dominican Republic. In the aftermath, figures from the dictator's administration took control, but under steady foreign pressure, gradually liberalized. But as the historian of modern Latin America Mark Healey has told me, even "the relatively soft post-Trujillo transition included significant bloodshed, a diplomatic crisis, multiple coups, and a US invasion". Nonetheless, when a Venezuelan friend tells me about his former students and colleagues who had been killed peacefully protesting Nicolás Maduro, I understand why he supports what we've done.

But when you launch military action in an unstable part of the world long beset by violence, there isn't really a bottom to how bad things can get. And rather than offering a serious plan for a stable future, we're already threatening other countries. President Donald Trump has revived talk about taking Greenland from our NATO ally, Denmark. When asked about possible military action in Colombia, another long-term ally and one whose assistance would be an essential part of any serious policy in the region, the president said, "Sounds good to me." 

Tough talk, but unlikely to engender trust in the region. And the recent seizures of tankers, along with the president's public statements suggesting a naked greed for oil, are certainly provoking moral repulsion.

As the analyst Kori Schake has argued, American power in the wake of World War II was remarkably cost-effective because it rested heavily on agreed-upon rules and consensual participation. "No dominant power," she wrote, "has ever had so much assistance from others in maintaining its dominance." If the only substitute the Trump administration has for getting willing cooperation from our allies is cooperation at the barrel of a gun, America will rapidly become a much diminished thing. Not just weaker, but also contemptible.

But it isn't even the potential downsides to Venezuela and the surrounding countries that worry me so much as what our president, high on the success of a brilliant military operation, might do next. In his first year back in office, we've already struck Iran and Venezuela without even trying to go through Congress and establish democratic support for war, the most dangerous and morally consequential thing a country can do, and which, for precisely that reason, our founders made the responsibility of the legislature.

Going forward, how often will Trump roll the dice this way? And if we continue to allow an unrestrained president to initiate acts of war against other countries without debate or authorization, how long before the whims of one irresponsible man lead to genuine disaster?
"""}

print("--- STARTING GENERATION (Thinking in Gray, Response in White) ---\n")

for chunk in chain.stream(inputs):
    # Check for reasoning (thinking) tokens
    # These are usually in additional_kwargs when reasoning=True
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if reasoning:
        print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
    
    # Check for the actual answer tokens
    content = chunk.content
    if content:
        print(content, end="", flush=True)

print("\n\n--- GENERATION COMPLETE ---")