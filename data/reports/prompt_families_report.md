# Implementation Scope Predictor — Prompt Family & Leakage-Safe Grouping Report

**Audit Date:** 2026-11-14  
**Total Examples Analyzed:** 869  
**Total Identified Groups:** 813  
**Singleton Groups (Unique Prompts):** 797  
**Multi-Member Prompt Families:** 16 (covering 72 prompts)

---

## 1. Grouping Methodology

To prevent data leakage during train/val/test splitting, prompts are grouped using a multi-stage semantic fingerprint:
1. **Identifier Normalization**: Synthetic identifiers (`scenario [0-9]+`, ticket references, Jira tags, entity IDs) are replaced with invariant tokens (`<SCENARIO_REF>`, `<TICKET_REF>`).
2. **Conversational Prefix Stripping**: Variable opening pleasantries (*"Hey"*, *"can u whip up"*, *"pls spin up"*) are normalized.
3. **Explicit Semantic Anchors**: High-risk template families (ServiceNow sync, AI project manager scenarios, Snowflake SQL API v2, Qt sensor rendering, React Native IAP, CSS tooltips) are mapped via explicit domain regexes.
4. **Sub-Phrase Character N-Gram Graph**: Cosine similarity on character 3-5 grams ($\ge 0.48$) is computed across normalized texts. Connected components form the final disjoint prompt families.

---

## 2. Group Size Distribution

| Group Size | Number of Groups | Total Prompts in Bin | Pct of Clean Dataset |
|---|---|---|---|
| **1 (Singletons)** | 797 | 797 | 91.7% |
| **2** | 10 | 20 | 2.3% |
| **3** | 3 | 9 | 1.0% |
| **9** | 1 | 9 | 1.0% |
| **10** | 1 | 10 | 1.2% |
| **24** | 1 | 24 | 2.8% |
| **Total** | **813** | **869** | **100.0%** |

---

## 3. All Multi-Member Prompt Families

### Family: `family_servicenow_sync` (Group ID: 1, Size: 24)
- **Row IDs:** `[1, 97, 121, 137, 257, 263, 266, 277, 280, 404, 410, 445, 504, 543, 545, 689, 693, 736, 771, 786, 810, 845, 887, 922]`
- **Primary Scopes:** `{'backend': 24}`
- **Representative Examples:**
  - `[Row 1]`: "Hey, need a quick python script to sync our incident tickets with the ServiceNow Table API. Should handle the sysparm_query encoding properly, do offset pagination since we have a lot of records, and make sure we don't duplicate entries if we run it twice (idempotent). Oh, and pls make sure it catches those rate limit errors (429) and structured API failures gracefully. Just need the backend logic for now, nothing fancy on the UI side."
  - `[Row 97]`: "I need a Python script to sync incidents from our ServiceNow instance. It should hit the Table API, handle the sysparm_query encoding properly, and do offset pagination since we have a lot of records. Also need it to be idempotent when updating records so we don't create dupes if it crashes halfway. Oh, and make sure it catches those rate limit 429s and retries correctly. Just a rough draft is fine for now, maybe call it snow_sync."
  - `[Row 121]`: "Hey, I need a quick python script to sync our incident tickets from ServiceNow. Just wanna pull records where the state is 'New' or 'In Progress', handle the pagination correctly (I think its offset based?), and make sure we don't dupes if we run it twice. Oh and if we hit rate limits, just wait and retry. Can you sketch out the logic for encoding the query params and handling the response envelope? Dont need a full app, just the core sync loop."

### Family: `family_ai_project_manager_scenarios` (Group ID: 132, Size: 10)
- **Row IDs:** `[160, 326, 347, 452, 530, 540, 641, 733, 764, 870]`
- **Primary Scopes:** `{'full_stack': 6, 'backend': 3, 'frontend': 1}`
- **Representative Examples:**
  - `[Row 160]`: "Hey, need to spin up a quick prototype for an AI project manager (internal ref: Scenario 762993). Just wanna get the core logic down in Python—stuff like task tracking, maybe some AI suggestions for breaking down epics, and saving everything to a DB. Don't worry about the frontend or auth yet, just the backend pieces to make it work. Can you sketch out how i'd structure the api and the data models? keep it simple for now."
  - `[Row 326]`: "hey, need to whip up a quick prototype for an ai project manager dashboard (internal scenario 645742). basically just the frontend UI for now – want a kanban board for task tracking and some analytics cards showing completion rates. dont worry about the backend logic or database yet, just make it look functional with some mock data so i can show the stakeholders tomorrow. oh, and make sure the color scheme matches our brand guidelines (dark mode). thx!"
  - `[Row 347]`: "Hey, need to spin up a quick prototype for an AI project manager, scenario ref 687116. Just wanna get the core stuff working: task tracking, some AI suggestions for breaking down goals, and basic analytics. Don't worry too much about the UI polish yet, just need it functional. Oh, and make sure users can collab on tasks. What's the best way to structure the backend and DB for this in Python? Maybe Flask or FastAPI?"

### Family: `family_snowflake_sql_api` (Group ID: 26, Size: 9)
- **Row IDs:** `[28, 44, 487, 655, 675, 744, 834, 916, 936]`
- **Primary Scopes:** `{'backend': 9}`
- **Representative Examples:**
  - `[Row 28]`: "I need to build a Python client for the Snowflake SQL API v2 that handles async queries properly. Before changing code, please check the official docs for the `/api/v2/statements` endpoint. The client needs to submit a statement with bound parameters, handle the 202 response by polling the status URL, and deal with a 408 timeout by preserving the statement handle. Also, make sure it retries on 429s but fails fast on 400s. Oh, and don't forget to include the required auth and content-type headers. I want to make sure we aren't leaking tokens in logs if a redirect happens."
  - `[Row 44]`: "Hey, I need to build a Python client for the Snowflake SQL API v2 to run some async queries in our data pipeline. We need to handle the requestId for idempotency, parse the bound parameters correctly, and deal with those 202/408 status codes when polling for results. Also need proper error reporting if the SQL fails. Can you sketch out the core logic for submitting a statement, polling the status URL, and handling the response partitions? Make sure to include the right headers and show how to retry if the connection drops but we have the request ID. Keep it stdlib only if possible, no heavy SDKs."
  - `[Row 487]`: "Hey, I need a quick python script to hit the Snowflake SQL API v2. The goal is to submit a query with `async=true`, then poll the status URL until it's done. Also need to handle the case where the request times out (408) but we still get a statement handle to check later. Oh, and make sure we use that requestId thing for idempotency in case the network flakes out. Just a simple runner, no fancy SDKs, standard lib only if possible. Let me know if you need the exact header details."

### Family: `family_qt_sensor_graphics` (Group ID: 2, Size: 3)
- **Row IDs:** `[2, 38, 74]`
- **Primary Scopes:** `{'full_stack': 2, 'frontend': 1}`
- **Representative Examples:**
  - `[Row 2]`: "Optimizing a Qt graphics view that renders thousands of sensor points but the UI freezes under load. Need to leverage threading and signal-slot architecture for responsiveness. Also the concurrent painting is causing artifacts. Maybe use OpenGL? The exact data rate varies so make it scalable. Just keep the interface smooth."
  - `[Row 38]`: "qt app is freezing when we pump thousands of sensor readings per second into the graphics view. using c++ and the signal-slot arch but the threading model seems off. need to optimize ui responsiveness under this heavy load. not sure if i should move the processing to a worker thread or change how the view updates, just stop the freeze."
  - `[Row 74]`: "qt graphics view freezing under heavy sensor load, threading model blocked in the c++ app. need to optimize ui responsiveness for the industrial viz system, probs a signal-slot bottleneck. fix the concurrent pattern so the desktop app stays smooth."

### Family: `family_react_native_iap` (Group ID: 186, Size: 3)
- **Row IDs:** `[224, 433, 859]`
- **Primary Scopes:** `{'full_stack': 2, 'frontend': 1}`
- **Representative Examples:**
  - `[Row 224]`: "react native in-app purchase flow is failing on android subscriptions, works fine on ios though. using rniap and firebase crashlytics but the async handling is messy. need to debug the edge cases where the payment hangs. not sure if its google play billing version or just a race condition, make it robust."
  - `[Row 433]`: "react native in-app purchase flow hangs on confirmation, fix the ios native module bridge. also the paywall screen css is ugly, make the buttons prominent and add some trust badges styling. make it convert. and handle restore purchases."
  - `[Row 859]`: "react native in-app purchase flow is breaking on android, subscription status not updating. firebase crashlytics shows errors in the async billing api. need to handle the edge cases for google play guidelines, optimize the ux so users dont drop off. fix the payment workflow pls."

### Family: `lexical_template_family_241` (Group ID: 241, Size: 3)
- **Row IDs:** `[288, 605, 924]`
- **Primary Scopes:** `{'backend': 3}`
- **Representative Examples:**
  - `[Row 288]`: "hey, i need to spin up a quick rest api in python to handle some user data. basically just endpoints to create and fetch users. oh and we gotta persist the stuff in a postgres db. can u sketch out the backend structure? dont worry about the frontend or auth for now, just the core logic and db connection. thnx"
  - `[Row 605]`: "Hey, I need to whip up a quick REST API in Python using Flask to handle some user data. Just the basics: endpoints to create a user and fetch their profile. Oh, and we gotta store this stuff in a Postgres DB. Make sure the auth is handled, maybe JWT? Also, the frontend team said the response needs to look 'clean' and 'modern' whatever that means lol. Can you sketch out the backend structure and maybe some DB schema? Dont worry about the actual UI code yet, just the API part."
  - `[Row 924]`: "hey, i need to whip up a quick rest api in python for a new microservice. basically just endpoints to create and fetch user profiles. im thinking flask or fastapi, whatever is faster to setup. oh and we gotta store the data in postgres. also make sure the auth is handled properly, maybe jwt? dont worry about the frontend yet, just the backend logic and db schema. keep it simple but secure."

### Family: `lexical_template_family_033` (Group ID: 33, Size: 2)
- **Row IDs:** `[36, 857]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 36]`: "hey, i need to whip up a rest api in python for our new microservice. it needs to handle user data CRUD ops and save everything to a postgres db. also make sure we got auth sorted out, prob oauth2 or jwt? oh and maybe hook it up to stripe for payments later if thats easy. just need the backend logic for now, frontend can wait. thx!"
  - `[Row 857]`: "hey, i need to whip up a quick REST API in python for our new microservice. it needs to handle user data CRUD ops and save everything to a postgres db. also make sure we got auth set up so random ppl cant hit it. oh and maybe connect it to that slack webhook for notifications? just need the backend logic mostly, frontend can wait. thx!"

### Family: `lexical_template_family_163` (Group ID: 163, Size: 2)
- **Row IDs:** `[198, 777]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 198]`: "pyserial code for reading industrial scales is dropping bytes, com port handling is flaky. interfacing with hardware in real-time and the data pipelining is broken. need to translate physical outputs to digital workflows reliably. maybe the baud rate config is wrong? just make the data stream consistent."
  - `[Row 777]`: "pyserial com port dropping bytes, industrial automation sensor data incomplete. need real-time processing fix for the hardware interface, probs a buffer flush issue. translate the physical device output to digital workflow reliably."

### Family: `family_sticky_footer_css` (Group ID: 177, Size: 2)
- **Row IDs:** `[213, 403]`
- **Primary Scopes:** `{'frontend': 2}`
- **Representative Examples:**
  - `[Row 213]`: "how do i make a footer that stays at the bottom of the page even if the content is short, but pushes down if the content is long? i've heard of sticky footers."
  - `[Row 403]`: "i want a sticky footer that stays at the bottom of the page even if the content is short. use a flex-grow approach so it doesn't overlap."

### Family: `family_tooltip_hover_css` (Group ID: 228, Size: 2)
- **Row IDs:** `[272, 770]`
- **Primary Scopes:** `{'frontend': 1, 'design_only': 1}`
- **Representative Examples:**
  - `[Row 272]`: "i need a a custom tooltip that appears when i hover over a word. it should be positioned relative to the cursor and have a little arrow pointing up."
  - `[Row 770]`: "i need a css-only tooltip that shows up when i hover over an element. it should be centered above the element and have a little arrow pointing down."

### Family: `lexical_template_family_346` (Group ID: 346, Size: 2)
- **Row IDs:** `[408, 681]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 408]`: "hey i need to whip up a simple rest api in python to handle user profiles. basically just endpoints to create and fetch a user. oh and we gotta save the data somewhere, maybe sqlite? not sure about the auth part yet, lets just get the basic crud working first. can u sketch out what files i'd need?"
  - `[Row 681]`: "Hey, need to whip up a quick REST API for a todo app using Flask. Just the basics: create, read, update, delete tasks. Oh, and we gotta save the data somewhere, maybe postgres? Not sure about the auth part yet, lets just get the endpoints working first. Keep it simple but make sure the db stuff is solid."

### Family: `lexical_template_family_466` (Group ID: 466, Size: 2)
- **Row IDs:** `[554, 866]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 554]`: "asp.net core middleware for trace headers is dropping spans in distributed systems. implementing observability and the http protocol handling is flawed. need to fix the tracing logic for cloud-native resilience. not sure if the correlation id generation is wrong or the header propagation, make it reliable."
  - `[Row 866]`: "asp.net core trace headers missing in distributed system, middleware not propagating ids. need observability init for the cloud native app, probs a http protocol gap. fix the tracing logic so the monitoring tools see the full request flow."

### Family: `lexical_template_family_475` (Group ID: 475, Size: 2)
- **Row IDs:** `[565, 881]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 565]`: "spring data mongodb inheritance patterns are broken, objects not mapping right to docs. need best practices for nosql data modeling in the scalable backend. fix the object-document mapping so the java app doesnt throw cast exceptions."
  - `[Row 881]`: "spring data mongodb inheritance pattern is broken. object-document mapping fails on polymorphic types. java code for data modeling is naive. need best practices for noSQL schema. app is crashing on save."

### Family: `lexical_template_family_543` (Group ID: 543, Size: 2)
- **Row IDs:** `[648, 943]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 648]`: "I've got a Python script that's acting weird with our shift handoff logs. First, the tags column won't go away even when I don't pass the `--show-tags` flag. It looks like the header grows every time I run it in the same process: `id | shift | notification | tags | tags | tags`. Works fine in a fresh shell, but breaks in our wrapper. Second, when I run `shiftlog show <bad_id>`, it prints an error message but exits with code 0, so our CI pipeline stays green even when the handoff reference is wrong. I need it to exit 1 on failure. Can you fix `shiftlog.py` so the tags column only appears when requested and the exit code is correct for missing IDs? Don't touch the tests; they're right."
  - `[Row 943]`: "Hey, I'm debugging this python CLI tool for our shift handoffs and it's acting super weird. The tags column won't go away even when I don't pass the flag, and if I run it in a loop the header just keeps growing tags tags tags... also when I look up a non-existent ID it prints an error but exits with code 0, which is breaking our CI pipeline. Can you fix shiftlog.py so the tags only show when requested and it actually exits 1 on missing IDs? Don't touch the tests, they're right."

### Family: `lexical_template_family_555` (Group ID: 555, Size: 2)
- **Row IDs:** `[664, 987]`
- **Primary Scopes:** `{'backend': 2}`
- **Representative Examples:**
  - `[Row 664]`: "hey can u help me setup a rest api in python to save user feedback data? need it to store the text and rating in a db. not sure if i need auth for this yet, maybe just a simple endpoint for now. also wanna make sure the design is clean but im mostly focused on the backend logic."
  - `[Row 987]`: "i need to setup a simple rest api in python to save some user feedback data. can u help me get the backend and db stuff going? not sure if i need auth yet maybe just a basic endpoint for now."

### Family: `lexical_template_family_622` (Group ID: 622, Size: 2)
- **Row IDs:** `[753, 833]`
- **Primary Scopes:** `{'full_stack': 2}`
- **Representative Examples:**
  - `[Row 753]`: "Hey, i need to whip up a quick script to save some user prefs to a DB. Maybe sqlite? Also wanna expose this via a REST endpoint so the frontend can hit it. Oh and make sure theres auth on the endpoint, dont want just anyone messing with the data. Keep the UI simple for now, just a basic form to test it out."
  - `[Row 833]`: "hey, i need to whip up a quick script to save some user prefs to a db. maybe sqlite? just a simple rest api endpoint too so the frontend can hit it. oh and make sure the ui looks decent, dark mode preferably. also add login stuff just in case we need it later? thx"

