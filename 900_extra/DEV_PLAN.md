# PLOVER: [P]robabilistic [L]anguage [O]ver [V]olumes for [E]nvironment [R]easoning
### In which I entertain an obsession with Geometric Grammars without knowledge of its Ultimate Value

# Design Principles && Slide Deck
* Relying on `(On Block Table)` as your robot's description of a rich and continuous reality should fill you with profound disappointment and unrelenting anger!
* Presenting: A representation with desirable properties
    - Scene graphs are maps that robots can plan on
    - Probabilistic scene graphs are maps that robots can plan on probabilistically
    - Desirable Properties, Clear and Correct representation of
        * Pose
        * Pose Uncert
        * Relationships
        * Relationship Uncert
* Facts should be physical first, and semantic second, (Cats Don't Study Semantics)            
    - Symbols (concepts) can be fuzzy
    - Predicates (relationships) can be fuzzy
        * Partially met
        * Uncertain
* How can a fact be physical? --> By expressing it geometrically!
    - When we can measure the degree to which a fact is true, We can optimize on degree 
        (Callback to degree of completion in Prelim)
    - When we can measure our confidence in the truth of a fact, We can optimize on confidence
* PDDL -vs- PLOVER Showdown: What it is like to solve the same problem in both frameworks?
    - Compare symbols
    - Compare predicates
    - Compare solver performance
        * Solver: Running time, Success rate
        * Execution: Running time, Success rate
* Explainability in Robot Plans
    - Geometric expression of facts has a side effect of being able to *render* facts to a display
    - If we can render facts, then we get visual explainability for (almost) free!
* Render the Physical Facts, Intuitively
    - Show facts from 2-Arches, and how they change over time
    - Show facts from Fruit Picking, and how they change over time
* Future Work
    - Allow the human to correct or create robot plans in a 3D env, Compare to MoveIt!
    - PLOVER planning in other metric spaces other than the physical (Word2Vec???)
        * Polytopes, distance, translation, and rotation (geo alg) all work the same in higher dims!
    - What is the connection to SLAM methods that operate on scene graphs?
    - What is the connection to VQA systems that operate on scene graphs?
        * Feed a visual scene to an LLM and let it explain the scene
    - Can PLOVER simplify working on hi-dim problems?
* `[ ]` **DANGER**: Review the PLOVER design principles at https://github.com/jwatson-CO-edu/CogArch/blob/main/README.md#plover-rich-real-world-representation

# `DEV PLAN`

## Representation, Computation, and Planner

* `[>]` Solve the 2-Arches problem in the 6-Block environemnt
    - `[>]` Belief
        * `[>]` Belief Samples
    - `[ ]` Required Symbols and Predicates (w/ PyBullet tests)
        * `[ ]` Block class, Using PLOVER as a lib
        * `[ ]` Block beliefs @ PyBullet
            - `{?}` Render belief? Ellipsoid?
        * `[ ]` Object @ Location predicate
            - `[ ]` Indicate in PyBullet
        * `[ ]` Object On Object predicate
            - `[ ]` Object Above Object predicate
            - `[ ]` Object Touching Object predicate
            - `[ ]` Execute handmade actions and allow system to identify all
            - `{?}` Object Supported by Object(s) predicate, WARNING: NOT MVP
        * `[ ]` Robot Holding Object predicate
    - `[ ]` Action Components
        * `[ ]` Lightweight Pre-Image Volume
            - `[ ]` Start with convex hull, This is a simple problem with few obstructions
            - `{?}` Move to Minkowski Sums only if needed
            - `[ ]` Render preimage to PyBullet
        * `[ ]` Define and Project a presumptive future state
            - `[ ]` Q: How to automatically sample targets that fulfill the req'd predicates?  Can this come directly from the predicate definition?
            - `[ ]` Q: How to differentiate a presumptive future fact from a present fact?
                * `[ ]` Q: Is there a need to establish durations in order to prevent collisions during planning?
            - `[ ]` Render presumptive state(s) to PyBullet      
    - `[ ]` Required Actions
        * `[ ]` Move Arm
            - `[ ]` Automatically resolve Move Free -vs- Move Holding w/ predicate
        * `[ ]` Pick
        * `[ ]` Place
            - `[ ]` Automatically resolve Stacked On/Touching relationships
    - `[ ]` Build Geometric Solver via increasingly complex problems, Execute Open Loop
        * Sample randomly. Do NOT optimize until it is ABSOLUTELY REQUIRED!
        * `[ ]` Object @ Location
            - `[ ]` Planned
            - `[ ]` Execute Open Loop
        * `[ ]` Object On Object
            - `[ ]` Planned
                * Sample in regions that will ground the desired predicate
            - `[ ]` Execute Open Loop
        * `[ ]` 1 Arch
            - `[ ]` Planned
                * `[ ]` Q: What is a general and efficient way to satisfy two `On` predicates simultaneously? 
                    Where to sample? How to compute where to samples based on {Predicates, Goals}?
            - `[ ]` Execute Open Loop
        * `[ ]` 2 Arches
            - `[ ]` Planned
                * `[ ]` Q: Does it make sense to automatically break a problem into subproblems?
                * `[ ]` Q: Is there an efficient means to determine if subproblems interfere with each other?
            - `[ ]` Execute Open Loop
    - `[ ]` Full TAMP Loop
        * `[ ]` Add physics gripper (increases credibility && complexity)
        * `[ ]` Collect data, esp. on solver performance
        * `[ ]` Demonstrate superiority over PDLS
    - `[ ]` Full MAGPIE Loop: MAGPIE and PLOVER are friends!

* `[ ]` Model the fruit picking problem: There are MANY questions to be answered!
    * Dream: System constructs a packed lattice of stacked interactions that allows fast planning on 
             single objects involving local interactions only
    - `[ ]` Q: Need to handle novel object classes?
        * `[ ]` Q: Can an object class remain indeterminate until it is identified with certainty?
    - `[ ]` Q: How to handle objects for which a model DNE?
        * `[ ]` Q: Does it make sense to *build* a model?
    - `[ ]` Q: Does this require shape completion?
    - `[ ]` Q: What are the LLM connections?
        * `[ ]` Q: Can geo predicates provide input for VQA? Can it talk about the geometrically identified relationships?
        * `[ ]` Q: Can the LLM suggest geo predicates?

* `[ ]` Model the ConMod problem
    - `[ ]` Q: What is the SIMPLEST way to model interaction with a granular medium?  Is sand an object?

## Show Them, Show Them All (They Called Me Mad)
- `[ ]` Assess Graduation Risk
- `[ ]` Refine PLOVER slide deck (above), **WARNING**: Having this as a backup proposal is mostly a distracting dream for entertainment!
- `[ ]` **RECENT** References
- `[ ]` Demo
    * `[ ]` Performant Planning
    * `[ ]` Intutive Output and Troubleshooting
    * `{?}` Human intervention?
- `[ ]` Choose audience
- `[ ]` Market PLOVER: Share what is exciting and true in a concise way

# Cool Features (DANGER: NON-MVP)
* `{?}` Render Predicates in 3D
* `{?}` Render Plan in 3D
* `{?}` Render Faults in 3D
* `{?}` JSON Text Description Language
    - `{?}` Q: How to be descriptive without containing ALL geo data?
    - `{?}` Predicates
    - `{?}` Actions
* `[ ]` DANGER: Review the PLOVER `DEV PLAN` at https://github.com/jwatson-CO-edu/CogArch/blob/main/README.md#plover-rich-real-world-representation
    - `[ ]` Q: How to model unintended/unmodeled side effects?
        * `[ ]` Q: How to even identify them?
* `{?}` Idea: Probabilistic constraints as preconditions for actions: e.g. 0.9 confidence of grasped object distribution not exceeding the max width of the volume between the gripper
    - `{?}` Render violated constraints as an explainable reason a plan was not executed
    - `{?}` Reason over the constraints least likely to have been met during a failed plan as possible explanations for failure
* `{?}` Idea: Use future durations to plan non-interfering sub-goals in parallel by declaring volumes occupied for that time
    - Multiple robots
    - Lazy, hierarchical planning at increasing granularity, planned in parallel
    - Assign entire dynamic regions of space their own planning process and/or robot
* `{?}` Idea: Learn probabilistic relationships between classes of objects so that we can begin searching for missing items based on scene information that we are confident in
* `{?}` Idea: Pair semantic regions with (likely) optimizers for "good" sampling in that region, e.g. Driving thru sandy terrain region should MP with kinodynamic RRT by default
* `{?}` Idea: Suggest grasp planners for object types