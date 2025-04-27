# from typing import Tuple
# from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
# from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
# from chains.custom_chains import (
#     get_summary_chain,
#     get_interests_chain,
#     get_ice_breaker_chain,
# )
# from third_parties.linkedin import scrape_linkedin_profile
# from third_parties.twitter import scrape_user_tweets, scrape_user_tweets_mock
# from output_parsers import (
#     Summary,
#     IceBreaker,
#     TopicOfInterest,
# )
#
#
# def ice_break_with(
#     name: str,
# ) -> Tuple[Summary, TopicOfInterest, IceBreaker, str]:
#     linkedin_username = linkedin_lookup_agent(name=name)
#     linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
#
#     twitter_username = twitter_lookup_agent(name=name)
#     tweets = scrape_user_tweets_mock(username=twitter_username)
#
#     summary_chain = get_summary_chain()
#     summary_and_facts: Summary = summary_chain.invoke(
#         input={"information": linkedin_data, "twitter_posts": tweets},
#     )
#
#     interests_chain = get_interests_chain()
#     interests: TopicOfInterest = interests_chain.invoke(
#         input={"information": linkedin_data, "twitter_posts": tweets}
#     )
#
#     ice_breaker_chain = get_ice_breaker_chain()
#     ice_breakers: IceBreaker = ice_breaker_chain.invoke(
#         input={"information": linkedin_data, "twitter_posts": tweets}
#     )
#
#     return (
#         summary_and_facts,
#         interests,
#         ice_breakers,
#         linkedin_data.get("photoUrl"),
#     )
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

information = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his leadership of Tesla, SpaceX, and X (formerly Twitter). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of March 2025, Forbes estimates his net worth to be US$345 billion. He was named Time magazine's Person of the Year in 2021.

Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He graduated from the University of Pennsylvania in the U.S. before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became a U.S. citizen.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence research but later left; growing discontent with the organization's direction in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. In January 2025, he was appointed head of Trump's newly created DOGE. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

Musk's political activities and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. Especially since the 2024 U.S. presidential election, Musk has been heavily involved in politics as a vocal supporter of Trump. Musk was the largest donor in the 2024 U.S. presidential election and is a supporter of global far-right figures, causes, and political parties. His role in the second Trump administration, particularly in regards to DOGE, has attracted public backlash.

Early life
See also: Musk family
Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[2][3] He is of British and Pennsylvania Dutch ancestry.[4][5] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[6][7][8][a] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[15][16][17][18] Elon has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[19][20][8][21] Musk was raised in the Anglican Church, in which he was baptized.[22][23]

The Musk family was wealthy during Elon's youth.[18] Despite both Elon and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[18] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[24][25] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[2]

After his parents divorced in 1980, Elon chose to live primarily with his father.[4][15] Elon later regretted his decision and became estranged from his father.[26] Elon has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations.[27] In one incident, after an altercation with a fellow pupil, Elon was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries.[28] Elon described his father berating him after he was discharged from the hospital.[28] Errol denied berating Elon and claimed, "The boy had just lost his father to suicide and Elon had called him stupid. Elon had a tendency to call people stupid. How could I possibly blame that child?"[29]

Elon was an enthusiastic reader of books, and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[17][30] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[31] At age twelve, Elon sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500.[32][33]

Education
An ornate school building
Musk graduated from Pretoria Boys High School in South Africa.
Musk attended Waterkloof House Preparatory School, Bryanston High School, and then Pretoria Boys High School, where he graduated.[34] Musk was a good but unexceptional student, earning a 61 in Afrikaans and a B on his senior math certification.[35] Musk applied for a Canadian passport through his Canadian-born mother to avoid South Africa's mandatory military service,[36][37] which would have forced him to participate in the apartheid regime,[2] as well as to ease his path to immigration to the United States.[38] While waiting for his application to be processed, he attended the University of Pretoria for five months.[39]

Musk arrived in Canada in June 1989, connected with a second cousin in Saskatchewan,[40][41] and worked odd jobs including at a farm and a lumber mill.[42] In 1990, he entered Queen's University in Kingston, Ontario.[43][44] Two years later, he transferred to the University of Pennsylvania, where he studied until 1995.[45] Although Musk has said that he earned his degrees in 1995, the University of Pennsylvania did not award them until 1997 – a Bachelor of Arts in physics and a Bachelor of Science in economics from the university's Wharton School.[46][47][48][49][50] He reportedly hosted large, ticketed house parties to help pay for tuition, and wrote a business plan for an electronic book-scanning service similar to Google Books.[51]

In 1994, Musk held two internships in Silicon Valley: one at energy storage startup Pinnacle Research Institute, which investigated electrolytic supercapacitors for energy storage, and another at Palo Alto–based startup Rocket Science Games.[52][53] In 1995, he was accepted to a graduate program in materials science at Stanford University, but did not enroll.[48][46][54] Musk decided to join the Internet boom, applying for a job at Netscape, to which he reportedly never received a response.[55][36] The Washington Post reported that Musk lacked legal authorization to remain and work in the United States after failing to enroll at Stanford.[54] In response, Musk said he was allowed to work at that time and that his student visa transitioned to an H1-B. According to numerous former business associates and shareholders, Musk said he was on a student visa at the time.[56]

Business career
Main article: Business career of Elon Musk
Zip2
Main article: Zip2
External videos
video icon Musk speaks of his early business experience during a 2014 commencement speech at University of Southern California on YouTube
In 1995, Musk, his brother Kimbal, and Greg Kouri founded web software company Zip2 with funds borrowed from Musk's father.[57][26] They housed the venture at a small rented office in Palo Alto.[58] The company developed and marketed an Internet city guide for the newspaper publishing industry, with maps, directions, and yellow pages.[59]

According to Musk, "The website was up during the day and I was coding it at night, seven days a week, all the time."[58] The Musk brothers obtained contracts with The New York Times and the Chicago Tribune,[60] and persuaded the board of directors to abandon plans for a merger with CitySearch.[61] Musk's attempts to become CEO were thwarted by the board.[62] Compaq acquired Zip2 for $307 million in cash in February 1999,[63][64] and Musk received $22 million for his 7-percent share.[65]

X.com and PayPal
Main articles: X.com (bank), PayPal, and PayPal Mafia
In 1999, Musk co-founded X.com, an online financial services and e-mail payment company.[66] The startup was one of the first federally insured online banks, and, in its initial months of operation, over 200,000 customers joined the service.[67] The company's investors regarded Musk as inexperienced and replaced him with Intuit CEO Bill Harris by the end of the year.[68] The following year, X.com merged with online bank Confinity to avoid competition.[58][68][69] Founded by Max Levchin and Peter Thiel,[70] Confinity had its own money-transfer service, PayPal, which was more popular than X.com's service.[71]

Within the merged company, Musk returned as CEO. Musk's preference for Microsoft software over Unix created a rift in the company and caused Thiel to resign.[72] Due to resulting technological issues and lack of a cohesive business model, the board ousted Musk and replaced him with Thiel in 2000.[73][b] Under Thiel, the company focused on the PayPal service and was renamed PayPal in 2001.[75][76] In 2002, PayPal was acquired by eBay for $1.5 billion in stock, of which Musk—the largest shareholder with 11.72% of shares—received $175.8 million.[77][78] In 2017, Musk purchased the domain X.com from PayPal for an undisclosed amount, stating that it had sentimental value.[79][80]

SpaceX
Main article: SpaceX

Musk explains Starship capabilities to leaders of North American Aerospace Defense Command, U.S. Northern Command, and Air Force Space Command in 2019
In 2001, Musk became involved with the nonprofit Mars Society and discussed funding plans to place a growth-chamber for plants on Mars.[81] Seeking a way to launch the greenhouse payloads into space, Musk made two unsuccessful trips to Moscow to purchase intercontinental ballistic missiles (ICBMs) from Russian companies NPO Lavochkin and Kosmotras. Musk instead decided to start a company to build affordable rockets.[82] With $100 million of his early fortune,[83] Musk founded SpaceX in May 2002 and became the company's CEO and Chief Engineer.[84][85]

SpaceX attempted its first launch of the Falcon 1 rocket in 2006.[86] Although the rocket failed to reach Earth orbit, it was awarded a Commercial Orbital Transportation Services program contract from NASA, then led by Mike Griffin.[87][88] After two more failed attempts that nearly caused Musk to go bankrupt,[86] SpaceX succeeded in launching the Falcon 1 into orbit in 2008.[89] Later that year, SpaceX received a $1.6 billion NASA contract for Falcon 9-launched Dragon spacecraft flights to the International Space Station (ISS), replacing the Space Shuttle after its 2011 retirement.[90] In 2012, the Dragon vehicle docked with the ISS, a first for a commercial spacecraft.[91]

Working towards its goal of reusable rockets, in 2015 SpaceX successfully landed the first stage of a Falcon 9 on a land platform.[92] Later landings were achieved on autonomous spaceport drone ships, an ocean-based recovery platform.[93] In 2018, SpaceX launched the Falcon Heavy; the inaugural mission carried Musk's personal Tesla Roadster as a dummy payload.[94][95] Since 2019,[96] SpaceX has been developing Starship, a reusable, super heavy-lift launch vehicle intended to replace the Falcon 9 and Falcon Heavy.[97] In 2020, SpaceX launched its first crewed flight, the Demo-2, becoming the first private company to place astronauts into orbit and dock a crewed spacecraft with the ISS.[98] In 2024, NASA awarded SpaceX an $843 million contract to deorbit the ISS at the end of its lifespan.[99]

Starlink
Main article: Starlink
See also: Starlink in the Russo-Ukrainian War

50 Starlink satellites shortly before deployment to low Earth orbit, 2019
In 2015, SpaceX began development of the Starlink constellation of low Earth orbit satellites to provide satellite Internet access.[100] After the launch of prototype satellites in 2018, the first large constellation was deployed in May 2019.[101] The total cost of the decade-long project to design, build, and deploy the constellation was estimated by SpaceX in 2020 to be $10 billion.[102][c]

During the Russian invasion of Ukraine, Musk provided free Starlink service to Ukraine, permitting Internet access and communication at a yearly cost to SpaceX of $400 million.[105][106][107][108][109] However, Musk refused to block Russian state media on Starlink.[110][111] In 2023, Musk denied Ukraine's request to activate Starlink over Crimea to aid an attack against the Russian navy, citing fears of a nuclear response.[112][113][114]

Tesla
Main article: Tesla, Inc.
Musks stands, arms crossed and grinning, before a Tesla Model S
Musk next to a Tesla Model S, 2011
Tesla, Inc., originally Tesla Motors, was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning. Both men played active roles in the company's early development prior to Musk's involvement.[115] Musk led the Series A round of investment in February 2004; he invested $6.35 million, became the majority shareholder, and joined Tesla's board of directors as chairman.[116][117] Musk took an active role within the company and oversaw Roadster product design, but was not deeply involved in day-to-day business operations.[118] Following a series of escalating conflicts in 2007, and the 2008 financial crisis, Eberhard was ousted from the firm.[119][page needed][120] Musk assumed leadership of the company as CEO and product architect in 2008.[121] A 2009 lawsuit settlement with Eberhard designated Musk as a Tesla co-founder, along with Tarpenning and two others.[122][123]

Tesla began delivery of the Roadster, an electric sports car, in 2008. With sales of about 2,500 vehicles, it was the first mass production all-electric car to use lithium-ion battery cells.[124] Under Musk, Tesla has since launched several well-selling electric vehicles, including the four-door sedan Model S (2012), the crossover Model X (2015), the mass-market sedan Model 3 (2017), the crossover Model Y (2020), and the pickup truck Cybertruck (2023).[125][126][127][128][129]

In May 2020 Musk resigned from chairman of the board as part of the settlement of a lawsuit from the SEC over him tweeting that funding had been "secured" for potentially taking Tesla private.[130][131]

The company has also constructed multiple lithium-ion battery and electric vehicle factories, called Gigafactories.[132] Since its initial public offering in 2010,[133] Tesla stock has risen significantly; it became the most valuable carmaker in summer 2020,[134][135] and it entered the S&P 500 later that year.[136][137] In October 2021, it reached a market capitalization of $1 trillion, the sixth company in U.S. history to do so.[138]

SolarCity and Tesla Energy
Main articles: SolarCity and Tesla Energy
Two green vans sporting the SolarCity logo
SolarCity solar-panel installation vans in 2009
Musk provided the initial concept and financial capital for SolarCity, which his cousins Lyndon and Peter Rive founded in 2006.[139] By 2013, SolarCity was the second largest provider of solar power systems in the United States.[140] In 2014, Musk promoted the idea of SolarCity building an advanced production facility in Buffalo, New York, triple the size of the largest solar plant in the United States.[141] Construction of the factory started in 2014 and was completed in 2017. It operated as a joint venture with Panasonic until early 2020.[142][143]

Tesla acquired SolarCity for $2 billion in 2016 and merged it with its battery unit to create Tesla Energy. The deal's announcement resulted in a more than 10% drop in Tesla's stock price; at the time, SolarCity was facing liquidity issues.[144] Multiple shareholder groups filed a lawsuit against Musk and Tesla's directors, stating that the purchase of SolarCity was done solely to benefit Musk and came at the expense of Tesla and its shareholders.[145][146] Tesla directors settled the lawsuit in January 2020, leaving Musk the sole remaining defendant.[147][148] Two years later, the court ruled in Musk's favor.[144]

Neuralink
Main article: Neuralink
Musk standing next to bulky medical equipment on a stage
Musk discussing a Neuralink device during a live demonstration in 2020
In 2016, Musk co-founded Neuralink, a neurotechnology startup, with an investment of $100 million.[149][150] Neuralink aims to integrate the human brain with artificial intelligence (AI) by creating devices that are embedded in the brain. Such technology could enhance memory or allow the devices to communicate with software.[150][151] The company also hopes to develop devices to treat neurological conditions like spinal cord injuries.[152] In 2022, Neuralink announced that clinical trials would begin by the end of the year.[153] In September 2023, the Food and Drug Administration approved Neuralink to initiate six-year human trials.[154]

Neuralink has conducted animal testing on macaques at the University of California, Davis. In 2021, the company released a video in which a macaque played the video game Pong via a Neuralink implant. The company's animal trials—which have caused the deaths of some monkeys—have led to claims of animal cruelty. The Physicians Committee for Responsible Medicine has alleged that Neuralink violated the Animal Welfare Act.[155] Employees have complained that pressure from Musk to accelerate development has led to botched experiments and unnecessary animal deaths. In 2022, a federal probe was launched into possible animal welfare violations by Neuralink.[156]

The Boring Company
Main article: The Boring Company
Musk speaks to a crowd of journalists. Behind him is a lighted tunnel.
Musk during the 2018 inauguration of the Boring test tunnel in Hawthorne, California
In 2017, Musk founded the Boring Company to construct tunnels, and revealed plans for specialized, underground, high-occupancy vehicles that could travel up to 150 miles per hour (240 km/h) and thus circumvent above-ground traffic in major cities.[157][158] Early in 2017, the company began discussions with regulatory bodies and initiated construction of a 30-foot (9.1 m) wide, 50-foot (15 m) long, and 15-foot (4.6 m) deep "test trench" on the premises of SpaceX's offices, as that required no permits.[159] The Los Angeles tunnel, less than two miles (3.2 km) in length, debuted to journalists in 2018. It used Tesla Model Xs and was reported to be a rough ride while traveling at suboptimal speeds.[160] Two tunnel projects announced in 2018, in Chicago and West Los Angeles, have been canceled.[161][162] However, a tunnel beneath the Las Vegas Convention Center was completed in early 2021.[163] Local officials have approved further expansions of the tunnel system.[164]

X Corp.
Main articles: Acquisition of Twitter by Elon Musk and Twitter under Elon Musk
Avatar of Elon Musk
Elon Musk 
@elonmusk
I made an offer
https://sec.gov/Archives/edgar...

April 14, 2022[165]
In early 2017, Musk expressed interest in buying Twitter and had questioned the platform's commitment to freedom of speech.[166][167][168] By 2022, Musk had reached 9.2% stake in the company,[169] making him the largest shareholder.[170][d] Musk later agreed to a deal that would appoint him to Twitter's board of directors and prohibit him from acquiring more than 14.9% of the company.[172][173] Days later, Musk made a $43 billion offer to buy Twitter.[170][174] By the end of April Musk had successfully concluded his bid for approximately $44 billion.[175] This included approximately $12.5 billion in loans and $21 billion in equity financing.[176][177] Having back tracked on his initial decision,[178] Musk bought the company on October 27, 2022.[179]

Immediately after the acquisition, Musk fired several top Twitter executives including CEO Parag Agrawal;[179][180] Musk became the CEO instead.[181] Under Elon Musk, Twitter instituted monthly subscriptions for a "blue check",[182][183][184] and laid off a significant portion of the company's staff.[185][186] Musk lessened content moderation and hate speech also increased on the platform after his takeover.[187][188][189][190] In late 2022, Musk released internal documents relating to Twitter's moderation of Hunter Biden's laptop controversy in the lead-up to the 2020 presidential election.[191] Musk also promised to step down as CEO after a Twitter poll,[192][193] and five months later, Musk stepped down from chief executive officer (CEO) and transitioned his role to executive chairman and chief technology officer (CTO).[194]

Despite Musk stepping down as CEO, X continues to struggle with challenges such as viral misinformation,[195] hate speech, and antisemitism controversies.[196][197] Musk has been accused of trying to silence some of his critics[198] by removing their accounts' blue checkmarks, which hinders visibility and is considered a form of shadow banning,[199][200] or suspending their accounts without justification.[201]

Other activities
Hyperloop
Main articles: Hyperloop and Hyperloop pod competition
A long white tube about 10 feet in diameter
A tube part of the 2017 Hyperloop pod competition, sponsored by SpaceX
In August 2013, Musk announced plans for a version of a vactrain, and assigned engineers from SpaceX and Tesla to design a transport system between Greater Los Angeles and the San Francisco Bay Area, at an estimated cost of $6 billion.[202][203] Later that year, Musk unveiled the concept, dubbed the Hyperloop,[204] intended to make travel cheaper than any other mode of transport for such long distances.[205]

OpenAI and xAI
Further information: OpenAI and xAI (company)
In December 2015, Musk co-founded OpenAI, a not-for-profit artificial intelligence (AI) research company aiming to develop artificial general intelligence, intended to be safe and beneficial to humanity.[206] Musk pledged $1 billion of funding to the company,[207] but only donated $50 million.[208] In 2018, Musk left the OpenAI board.[209] Since 2018, OpenAI has made significant advances in machine learning.[210] In July 2023, Musk launched the artificial intelligence company xAI, which aims to develop a generative AI program that competes with existing offerings like OpenAI's ChatGPT. Musk obtained funding from investors in SpaceX and Tesla,[211] and xAI hired engineers from Google and OpenAI.[212]

Private jet
Main articles: ElonJet and 2022 Twitter suspensions
Avatar of Elon Musk
Elon Musk
@elonmusk
Same doxxing rules apply to "journalists" as to everyone else

December 16, 2022[213]
Musk uses a private jet owned by Falcon Landing LLC, a SpaceX-linked company, and acquired a second jet in August 2020.[214][215] His heavy use of the jets and the consequent fossil fuel usage have received criticism.[214][216] Musk's flight usage is tracked on social media through ElonJet.[217][218][219] In December 2022, Musk banned the ElonJet account on Twitter, as well as temporary bans on the accounts of journalists that posted stories regarding the incident, including Donie O'Sullivan, Keith Olbermann, and journalists from The New York Times, The Washington Post, CNN, and The Intercept.[220]

Politics
Main article: Political activities of Elon Musk
See also: Protests against Elon Musk

Musk with then-president-elect Donald Trump in November 2024
Musk is an outlier among business leaders who typically avoid partisan political advocacy.[221][222][223] Musk was a registered independent voter when he lived in California.[224] Historically, he has donated to both Democrats and Republicans,[225] many of whom serve in states in which he has a vested interest.[226] Since 2022, his political contributions have mostly supported Republicans, with his first vote for a Republican going to Mayra Flores in the 2022 Texas's 34th congressional district special election.[227][228] In 2024, he started supporting international far-right political parties, activists, and causes,[229] and has shared far-right misinformation[230][231][232] and numerous conspiracy theories.[233][234] Since 2024, his views have been generally described as right-wing.[235]

Musk supported Barack Obama in 2008 and 2012,[236] Hillary Clinton in 2016, Joe Biden in 2020,[237] and Donald Trump in 2024.[238] In the 2020 Democratic Party presidential primaries, Musk endorsed candidate Andrew Yang and expressed support for Yang's proposed universal basic income,[239] and endorsed Kanye West's 2020 presidential campaign.[240] In 2021, Musk publicly expressed opposition to the Build Back Better Act, a $3.5 trillion legislative package endorsed by Joe Biden that ultimately failed to pass due to unanimous opposition from congressional Republicans and several Democrats.[241] In 2022, Musk said he would start supporting Republican Party candidates,[242] and gave over $50 million to Citizens for Sanity, a conservative political action committee.[243] In 2023, he supported Republican Ron DeSantis for the 2024 U.S. presidential election, giving $10 million to his campaign,[243] and hosted DeSantis's campaign announcement on a Twitter Spaces event.[244][245][246] From June 2023 to January 2024, Musk hosted a bipartisan set of X Spaces with Republican and Democratic candidates, including Robert F. Kennedy Jr.,[247] Vivek Ramaswamy,[248] and Dean Phillips.[249]


Musk at a 2024 gathering with Trump and other political leaders
By early 2024, Musk had become a vocal and financial supporter of Donald Trump.[250] In July 2024, minutes after the attempted assassination of Donald Trump, Musk endorsed him for president.[251][252] During the presidential campaign, Musk joined Trump on stage at a campaign rally,[253] and during the campaign promoted conspiracy theories and falsehoods about Democrats, election fraud[254] and immigration, in support of Trump.[255][256] Musk was the largest individual donor of the 2024 election.[257] In 2025, Musk contributed $19 million to the Wisconsin Supreme Court race, hoping to influence the state's future redistricting efforts and its regulations governing car manufacturers and dealers.[258][259]

Musk's international political actions and comments have come under increasing scrutiny and criticism, especially from the governments and leaders of France, Germany, Norway, Spain and the United Kingdom, particularly due to his position in the U.S. government as well as ownership of X.[260][261][262] An NBC News analysis found he had boosted far-right political movements to cut immigration and curtail regulation of business in at least 18 countries on six continents since 2023.[263]

Trump's inauguration
Main article: Elon Musk salute controversy

Musk giving a gesture at the second inauguration of Donald Trump before saying "My heart goes out to you. It is thanks to you that the future of civilization is assured."[264][265]
In his speech during the second inauguration of Donald Trump, Musk thumped his right hand over his heart, fingers spread wide, and then extended his right arm out, emphatically, at an upward angle, palm down and fingers together. He then repeated the gesture to the crowd behind him. As he finished the gestures, he said to the crowd, "My heart goes out to you. It is thanks to you that the future of civilization is assured."[264][265][266] The gesture was viewed as a Nazi or Roman salute[e] by some.[267][268][269] Musk derided the claims as politicized.[270][271] In a social media post, he wrote: "The 'everyone is Hitler' attack is sooo tired",[272] and has since denied it.[273] In further response to the events, Musk posted a series of puns about Nazis on Twitter.[274] Various media outlets, including the Associated Press, reported that regardless of what Musk meant, his gesture was widely embraced by right-wing extremists and neo-Nazis.[275][276]

Department of Government Efficiency
Main article: Department of Government Efficiency

Elon Musk wielding a chainsaw at the Conservative Political Action Conference (CPAC) in 2025, imitating a publicity stunt used by Javier Milei symbolic of efficiency, federal mass layoffs and tax cutting
The concept of DOGE emerged in a discussion between Musk and Donald Trump, and in August 2024, Trump committed to giving Musk an advisory role, with Musk accepting the offer.[277] In November and December 2024, Musk suggested that the organization could help to cut the U.S. federal budget, consolidate the number of federal agencies,[278][279] and eliminate the Consumer Financial Protection Bureau,[280][281] and that its final stage would be "deleting itself".[282]

In January 2025, the organization was created by executive order, and Musk was designated a "special government employee".[283][284] Musk is leading the organization and is a senior advisor to the president,[285] although his official role is not clear.[286] In sworn statement during a lawsuit, the director of the White House Office of Administration stated that Musk "is not an employee of the U.S. DOGE Service or U.S. DOGE Service Temporary Organization", "is not the U.S. DOGE Service administrator", and has "no actual or formal authority to make government decisions himself".[287][288] Trump said two days later that he had put Musk in charge of DOGE.[289] A federal judge has ruled that Musk acts as the de facto leader of DOGE.[290]

In early 2025, Musk was criticized for his treatment of federal government employees,[291][292][293] including his influence over the mass layoffs of the federal workforce.[294][295][296] He has prioritized secrecy within the organization[297] and has accused others of violating privacy laws.[283]

Views
Main article: Views of Elon Musk
Avatar of Elon Musk
Elon Musk
@elonmusk
My commitment to free speech extends even to not banning the account following my plane, even though that is a direct personal safety risk

November 6, 2022[298]
Rejecting the conservative label,[299] Musk has described himself as a political moderate, even as his views have become more right-wing over time.[300] His views have been characterized as libertarian and far-right,[301][302] and after his involvement in European politics, they have received criticism from world leaders such as Emmanuel Macron and Olaf Scholz.[303][304][305][306]

Within the context of American politics, Musk supported Democratic candidates up until 2022, at which point he voted for a Republican for the first time.[236][242][238] He has stated support for universal basic income,[307] gun rights,[308] freedom of speech,[309] a tax on carbon emissions,[310] and H-1B visas.[311] Musk has expressed concern about issues such as artificial intelligence (AI)[312] and climate change,[313] and has been a critic of wealth tax,[314] short-selling,[315] government subsidies.[316] An immigrant himself, Musk has been accused of being anti-immigration, and regularly blames immigration policies for illegal immigration.[317] He is also a pronatalist who believes population decline is the biggest threat to civilization,[318] and believes in the principles of Christianity.[319][320] Musk has long been an advocate for space colonization, especially the colonization of Mars. He has repeatedly pushed for humanity colonizing Mars, in order to become an interplanetary species and lower the risks of human extinction.[321]

Musk has promoted conspiracy theories and made controversial statements that have led to accusations of racism, sexism, antisemitism,[322][323] transphobia,[324] disseminating disinformation, and support of white pride.[325][326] While describing himself as a "pro-Semite",[327] his comments regarding George Soros and Jewish communities have been condemned by the Anti-Defamation League and the White House.[328] Musk was criticized during the COVID-19 pandemic for making unfounded epidemiological claims,[329] defied COVID-19 lockdowns restrictions,[330] and supported the Canada convoy protest against vaccine mandates.[331][332]

International relations
Main article: International relations of Elon Musk

Musk with the president of Israel Isaac Herzog, November 2023
Musk has been critical of Israel's actions in the Gaza Strip during the Gaza war,[333] praised China's economic and climate goals,[334][335] suggested that Taiwan and China should resolve cross-strait relations,[336][337] and was described as having a close relationship with the Chinese government.[334][335]

In Europe, Musk expressed support for Ukraine in 2022 during the Russian invasion, recommended referendums and peace deals on the annexed Russia-occupied territories,[338][339] and supported the far-right Alternative for Germany in Germany in 2024.[340] Regarding British politics, Musk blamed the 2024 UK riots on mass migration and open borders,[341][342] criticized Prime Minister Keir Starmer for what he described as a "two-tier" policing system,[343][344][342] and was subsequently attacked as being responsible for spreading misinformation and amplifying the far-right.[345] He has also voiced his support for far-right activist Tommy Robinson and pledged electoral support for Reform UK.[346][347]

Legal affairs
Main article: Legal affairs of Elon Musk
Further information: List of lawsuits involving Tesla, Inc. and Criticism of Tesla, Inc.
In 2018, Musk was sued by the U.S. Securities and Exchange Commission (SEC) for a tweet stating that funding had been secured for potentially taking Tesla private.[130][f] The securities fraud lawsuit characterized the tweet as false, misleading, and damaging to investors, and sought to bar Musk from serving as CEO of publicly traded companies.[130][351][352] Two days later, Musk settled with the SEC, without admitting or denying the SEC's allegations. As a result, Musk and Tesla were fined $20 million each, and Musk was forced to step down for three years as Tesla chairman but was able to remain as CEO.[131] Shareholders filed a lawsuit over the tweet,[353] and in February 2023, a jury found Musk and Tesla not liable.[354] Musk has stated in interviews that he does not regret posting the tweet that triggered the SEC investigation.[355][356]

In 2019, Musk stated in a tweet that Tesla would build half a million cars that year.[357] The SEC reacted by asking a court to hold him in contempt for violating the terms of the 2018 settlement agreement. A joint agreement between Musk and the SEC eventually clarified the previous agreement details,[358] including a list of topics about which Musk needed preclearance.[359] In 2020, a judge blocked a lawsuit that claimed a tweet by Musk regarding Tesla stock price ("too high imo") violated the agreement.[360][361] Freedom of Information Act (FOIA)-released records showed that the SEC concluded Musk had subsequently violated the agreement twice by tweeting regarding "Tesla's solar roof production volumes and its stock price".[362]

In October 2023, the SEC sued Musk over his refusal to testify a third time in an investigation into whether he violated federal law by purchasing Twitter stock in 2022.[363][364][365] In February 2024, Judge Laurel Beeler ruled that Musk must testify again.[366] In January 2025, the SEC filed a lawsuit against Musk for securities violations related to his purchase of Twitter.[367] In January 2024, Delaware judge Kathaleen McCormick ruled in a 2018 lawsuit that Musk's $55 billion pay package from Tesla be rescinded.[368] McCormick called the compensation granted by the company's board "an unfathomable sum" that was unfair to shareholders.[369]

Personal life
Musk became a U.S. citizen in 2002.[45] From the early 2000s until late 2020, Musk resided in California, where both Tesla and SpaceX were founded.[370] He then relocated to Cameron County, Texas,[371][372] saying that California had become "complacent" about its economic success.[370][373][374]

While hosting Saturday Night Live in 2021, Musk stated that he has Asperger syndrome (now merged with autism spectrum disorder), although he has not been formally diagnosed.[375][376] Musk suffers from back pain and has undergone several spine-related surgeries, including a disc replacement.[377][378] In 2000, he contracted a severe case of malaria while on vacation in South Africa.[379] Musk has stated he uses doctor-prescribed ketamine for occasional depression and that he doses "a small amount once every other week or something like that";[380] since January 2024, some media outlets have reported that he takes ketamine, marijuana, LSD, ecstasy, mushrooms, cocaine and other drugs. Musk at first refused to comment on his alleged drug use, before responding that he had not tested positive for drugs, and that if drugs somehow improved his productivity, "I would definitely take them!".[381]

Through his own label Emo G Records, Musk released a rap track, "RIP Harambe", on SoundCloud in March 2019.[382][383][384] The following year, he released an EDM track, "Don't Doubt Ur Vibe", featuring his own lyrics and vocals.[385]

Musk plays video games, which he stated has a "'restoring effect' that helps his 'mental calibration'".[386] Some games he plays include Quake, Diablo IV, Elden Ring, and Polytopia.[387][388] Musk once claimed to be one of the world's top video game players but has since admitted to "account boosting", or cheating by hiring outside services to achieve top player rankings.[389][390][391] Musk has justified the boosting by claiming that all top accounts do it so he has to as well to remain competitive.[392][391][393] In 2024 and 2025, Musk criticized the video game Assassin's Creed Shadows and its creator Ubisoft for "woke" content.[394] Musk posted to X that "DEI kills art" and specified the inclusion of the historical figure Yasuke in the Assassin's Creed game as offensive; he also called the game "terrible". Ubisoft responded by saying that Musk's comments were "just feeding hatred" and that they were focused on producing a game not pushing politics.[395][396]

Relationships and children
Further information: Musk family

Musk with his son, X Æ A-Xii, in the Oval Office, February 2025
Musk has fathered at least fourteen children, one of whom died as an infant.[397] He had six children with his first wife, Canadian author Justine Wilson, who he met while attending Queen's University in Ontario, Canada; they married in 2000.[398] In 2002, their first child Nevada Musk died of sudden infant death syndrome at the age of 10 weeks.[399] After his death, the couple used in vitro fertilization (IVF) to continue their family;[400] they had twins in 2004, followed by triplets in 2006.[400] The couple divorced in 2008 and have shared custody of their children.[401][402] The elder twin he had with Wilson came out as a trans woman and, in 2022, officially changed her name to Vivian Jenna Wilson,[403] adopting her mother's surname because she no longer wished to be associated with Musk.[403]

Musk began dating English actress Talulah Riley in 2008.[404] They married two years later at Dornoch Cathedral in Scotland.[405][406] In 2012, the couple divorced, before remarrying the following year.[407] After briefly filing for divorce in 2014,[407] Musk finalized a second divorce from Riley in 2016.[408] Musk then dated Amber Heard for several months in 2017;[409] he had reportedly been "pursuing" her since 2012.[410]

In 2018, Musk and Canadian musician Grimes confirmed they were dating.[411] Grimes and Musk have three children, born in 2020, 2021, and 2022.[412][413][414][415] Musk and Grimes originally gave their eldest child the name "X Æ A-12", which would have violated California regulations as it contained characters that are not in the modern English alphabet;[416][417] the names registered on the birth certificate are "X" as a first name, "Æ A-Xii" as a middle name, and "Musk" as a last name.[418][419] They received criticism for choosing a name perceived to be impractical and difficult to pronounce;[420] Musk has said the intended pronunciation is "X Ash A Twelve".[419] Their second child was born via surrogacy.[421] Despite the pregnancy, Musk confirmed reports that the couple were "semi-separated" in September 2021; in an interview with Time in December 2021, he said he was single.[422][423] In October 2023, Grimes sued Musk over parental rights and custody of X Æ A-Xii.[424][425][426]

Musk also has four children with Shivon Zilis, director of operations and special projects at Neuralink: twins born via IVF in 2021, a child born in 2024 via surrogacy and a child born in 2025.[427][428][429][430][431] Musk allegedly had a child with author Ashley St. Clair in 2024.[432][1] The Wall Street Journal reported that sources close to Musk suggest that the "true number of Musk's children is much higher than publicly known".[1]

Wealth
These paragraphs are an excerpt from Wealth of Elon Musk.[edit]
Elon Musk is the wealthiest person in the world, with an estimated net worth of US$330 billion as of March 2025, according to the Bloomberg Billionaires Index,[433] and $359.5 billion according to Forbes,[434] primarily from his ownership stakes in Tesla and SpaceX.[435]

Having been first listed on the Forbes Billionaires List in 2012,[436] around 75% of Musk's wealth was derived from Tesla stock in November 2020.[437] Describing himself as "cash poor",[438][439] he became the first person in the world to have a net worth above $300 billion a year later. By December 2024, he became the first person to reach a net worth of $400 billion.[440]
Musk Foundation
Main article: Musk Foundation
Musk is president of the Musk Foundation he founded in 2001,[441][442] whose stated purpose is to provide solar-power energy systems in disaster areas, with an interest in human space exploration, pediatrics, renewable energy, and "safe artificial intelligence".[443] From 2002 to 2018, the foundation donated nearly half of its $25 million directly to Musk's OpenAI.[444][445] The foundation's assets reached $9.4 billion by the end of 2021, but it only dispensed $160 million to charities that year.[446]

The Musk Foundation has been criticized for its "self-serving"[446] donations to efforts close to Musk's family and companies,[447] as well as its low payout ratio.[446][448] In 2021, after Musk challenged World Food Programme director David Beasley to draft a plan to use money of Musk's that Beasley said could contribute to ending world hunger,[449][450] Musk instead donated the $6 billion in question to his own foundation even after Beasley's plan showed that the money could feed 42 million people for a year.[451][446][452]

Public image
Main article: Public image of Elon Musk
Musk in the Oval Office with Trump, 2025
Although his ventures have been highly influential within their separate industries starting in the 2000s, Musk only became a public figure in the early 2010s. He has been described as an eccentric who makes spontaneous and impactful decisions, while also often making controversial statements, contrary to other billionaires who prefer reclusiveness to protect their businesses. Musk's actions and his expressed views have made him a polarizing figure.[453] Biographer Ashlee Vance described people's opinions of Musk as polarized due to his "part philosopher, part troll" persona on Twitter.[454]

Musk has been described as an American oligarch due to his extensive influence over public discourse, social media, industry, politics, and government policy.[455] After Trump's re-election, Musk's influence and actions during the transition period and the second presidency of Donald Trump led some to call him "President Musk", the "actual president-elect", "shadow president" or "co-president".[456][457]

Accolades
Main article: List of awards and honors received by Elon Musk
Musk wearing a medal
Musk receiving the Order of Defence Merit from the Brazilian Armed Forces in 2022[458]
Awards for his contributions to the development of the Falcon rockets include the American Institute of Aeronautics and Astronautics George Low Transportation Award in 2008,[459] the Fédération Aéronautique Internationale Gold Space Medal in 2010,[460] and the Royal Aeronautical Society Gold Medal in 2012.[461] In 2015, he received an honorary doctorate in engineering and technology from Yale University[462] and an Institute of Electrical and Electronics Engineers Honorary Membership.[463] Musk was elected a Fellow of the Royal Society (FRS) in 2018.[464][g] In 2022, Musk was elected to the National Academy of Engineering.[466]

Time has listed Musk as one of the most influential people in the world in 2010,[467] 2013,[468] 2018,[469] and 2021.[470] Musk was selected as Time's "Person of the Year" for 2021. Then Time editor-in-chief Edward Felsenthal wrote that, "Person of the Year is a marker of influence, and few individuals have had more influence than Musk on life on Earth, and potentially life off Earth too."[471][472]

In popular culture
See also: Elon Musk filmography
Musk was a partial inspiration for the characterization of Tony Stark in the Marvel film Iron Man (2008).[473] Musk also had a cameo appearance in the film's 2010 sequel, Iron Man 2.[474] Other films that he has made cameos and appearances include Machete Kills (2013),[475] Why Him? (2016),[476] and Men in Black: International (2019).[477] Television series in which he has appeared include The Simpsons ("The Musk Who Fell to Earth", 2015),[478] The Big Bang Theory ("The Platonic Permutation", 2015),[479] South Park ("Members Only", 2016),[480][481] Young Sheldon ("A Patch, a Modem, and a Zantac®", 2017),[482] Rick and Morty ("One Crew over the Crewcoo's Morty", 2019).[483][484] He contributed interviews to the documentaries Racing Extinction (2015) and Lo and Behold (2016).[485][486]
"""

if __name__ == "__main__":
    print("Welcome to Ice Breaker")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    # llm = ChatOllama(model="llama3.2")

    chain = summary_prompt_template | llm | StrOutputParser()
    print("CHAIN: \n", chain)

    res = chain.invoke(input={"information": information})

    print(res)
