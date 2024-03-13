# PLOVER: [P]robabilistic [L]anguage [O]ver [V]olumes for [E]nvironment [R]easoning
### In which I entertain an obsession with Geometric Grammars without knowledge of its Ultimate Value

# Design Principles && Slide Deck
* The current, fastest route to a general purpose robot is **T**ask **A**nd **M**otion **P**lanning! (TAMP)
* Relying on `(On Block Table)`, as TAMP does, for your robot's description of a rich and continuous reality should fill you with profound disappointment and unrelenting anger!
* Presenting: A representation with desirable properties
    - Desirable Properties, Clear and Correct representation of
        * Continuous Pose
        * Pose Uncert
        * Continuous *Relationships*
        * Relationship Uncert
    - Scene graphs are maps that robots can plan on
    - Probabilistic scene graphs are maps that robots can plan on probabilistically
* Facts should be physical first, and semantic second, (Cats Don't Study Semantics)            
    - Symbols (concepts) can be fuzzy
    - Predicates (relationships) can be fuzzy
        * Partially met
        * Uncertain
    - Symbols should have geometric properties that makes their evaluation lightweight, intuitive to humans, and relevant to solving physical problems.
    - Composable graphs of geometric symbols can represent complex, context-sensitive relationships
* How can a fact be physical? --> By expressing it geometrically!
    - When we can measure the degree to which a fact is true, We can optimize on degree 
        (Callback to degree of completion in Prelim)
    - When we can measure our confidence in the truth of a fact, We can optimize on confidence
    - Scene graphs naturally model **context** with nested structure
* PDDL -vs- PLOVER Showdown: What it is like to solve the same problem in both frameworks?
    - Compare symbols
    - Compare predicates
    - Compare solver performance
        * Solver: Running time, Success rate
        * Execution: Running time, Success rate
* Explainability in Robot Plans
    - Geometric expression of facts has a side effect of being able to *render facts* to a display
    - If we can render facts, then we get *visual explainability* for (almost) free!
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

## Development Strategy
1. Build the system by solving the simplest problem(s) **first**!
    * **Test** as you go.
        1. 1-Arch
        1. 2-Arch
        1. Fruit Basket
        1. ConMod
    * Specific **first**, General _later_
        - Serially solving specific problems with testable goals will _gradually_ reveal which generalities the system actually **needs**.
        - Your initial impressions about universality will _limit_ your options more often than _expanding_ them
1. **M**inimum **V**iable **P**rogram ONLY!
    * Add features/complexity only where **absolutely necessary**!
    * Discard approaches that are slowing you down.  
    <u>Avoid</u>:
        - Solving problems that _do not yet exist_
        - Parallel programming
1. Platform
    1. Prototype in Python for ease. Python will drive adoption.
    1. Stay *independent* of ROS, at least until it is applied to a physical robotics problem.
    1. Only consider a compiled language if performance is a recurring bottleneck
    

## Representation, Computation, and Planning

* `[>]` Solve the 1-Arch problem in the 6-Block environemnt
    - `[Y]` Belief, 2024-02-24: Written (Ported)
        * `[Y]` Belief Samples, 2024-02-24: Written (Ported)
    - `[Y]` Required Symbols (w/ PyBullet tests), 2024-03-12: Colors are INTS on [0,255]
        * `[Y]` Block representation, Using PLOVER as a lib, 2024-02-26: Pose sampling is now sane!
            - 2024-02-25: Wrote function + lookup instead of a `class`.
            - Should **not** have to write a class for every new object, as this burdens the practitioner and does not support flexibility
            - Rather, provide the system with a way to associate labels with `Volume`s, whether it looks them up or constructs them from observation
        * `[Y]` Block beliefs @ PyBullet, 2024-03-12: Colors are INTS on [0,255]
            - 2024-02-27: I would like to avoid tight coupling with the world present in the current implementation of MAGPIE
            - `[Y]` Render Sample, 2024-03-12: Colors are INTS on [0,255]
                * `[Y]` Transform mesh, 2024-03-12: Colors are INTS on [0,255]
                * `[Y]` Render meshes from a complete scan, 2024-03-12: Colors are INTS on [0,255]
            - `[Y]` Render belief, 2024-03-12: Colors are INTS on [0,255]
                * `[Y]` Draw with transparency, 2024-03-12: Colors are INTS on [0,255]
                * `[Y]` Position ellipsoid, 2024-03-12: Colors are INTS on [0,255]
                * `[Y]` Orientation sticks, 2024-03-12: Colors are INTS on [0,255]
    - `[>]` Beliefs are part of the scene graph
    - `[>]` Required Predicates
        * `[>]` Predicate Base Class
        * `[ ]` Object @ Location predicate, with _probability_
            - `[ ]` Indicate in PyBullet
        * `[ ]` Object On Object predicate, with _probability_
            - `[ ]` Object Above Object predicate
            - `[ ]` Object Touching Object predicate
            - `[ ]` Execute handmade actions and allow system to identify all
            - `{?}` Object Supported by Object(s) predicate, WARNING: NOT MVP
        * `[ ]` Robot Holding Object predicate
    - `[ ]` Action Components
        * `[ ]` Lightweight Pre-Image Volume
            - `[ ]` Start with convex hull, This is a simple problem with few obstructions
                * This *might* not work with a gripper!
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

* `[ ]` Solve the 2-Arch problem in the 6-Block environemnt
    - `[ ]` Compare planner performance with current MAGPIE, per-action: {Success Rate, Computation Time}
    - `[ ]` Compare execution performance with current MAGPIE: {Success Rate, Makespan}

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
        * `{?}` Idea: Terrain is a special object with a volumetric representation that expands with problem context 

## Demo (SUSPENDED)
### You may not be able to convince your committee this is your Thesis!
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
### Please, please do not attempt these! It is likely their pursuit would jeopardize your education and/or career!  
### They are **not** required for the most basic Minimum Viable Program implementation of the PLOVER concept!
* `{?}` Render Predicates in 3D
* `{?}` Render Plan in 3D
* `{?}` Render Faults in 3D
* `{?}` JSON Text Description Language
    - `{?}` Q: How to be descriptive without containing ALL geo data?
    - `{?}` Predicates
    - `{?}` Actions
* `{?}` Evaluate:
    - `{?}` ROS package
    - `{?}` MoveIt! integration
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
* `{?}` Idea: Allow disjoint scene graphs
    - As a temporary solution to the Kidnapped Robot Problem. Can be merged later through exploration?
    - To allow the solver to work on multiple problems?

# Future Work (EXTREME DANGER: CAREER THREATS)
* How to integrate Configuration Space representations?
* `{?}` Idea: Learn probabilistic relationships between classes of objects so that we can begin searching for missing items based on scene information that we are confident in
* `{?}` Idea: Pair semantic regions with (likely) optimizers for "good" sampling in that region, e.g. Driving thru sandy terrain region should MP with kinodynamic RRT by default
* `{?}` Idea: Suggest grasp planners for object types
* `{?}` Idea: The system constructs label meshes from observation and interaction (from simple to complex)
    - Visual Content:
        1. Flat, Uniform color
        1. Textured meshes
    - Construction:
        1. Specfic: 1 to 1 label to mesh mapping
        1. Exemplar: Each label can be represented by any of a collection of meshes
        1. Prototype: System can match observations to a parameterized model of the class of labels
        1. Learned: Generative NN model of each label both classifies (and completes) observations
* `{?}` Idea: Optimize samples in the context of the current problem as background process(es), Cache results
    - Robot motion roadmap as a network of scored configs
