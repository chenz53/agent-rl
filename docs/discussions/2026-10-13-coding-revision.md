# Coding-first tutorial revision

Recorded 2026-10-13. English summary of the learner's Chinese instructions, not a verbatim English quotation.

The learner requested: replace the domain case with coding questions, start from DeepSWE, define each teaching function from scratch instead of importing nanoAstra helpers, correct the disconnected order between chapters 1 and 2, place display-math delimiters on separate lines, and use helpful visualizations.

The tutor inspected the public DeepSWE report and pinned rLLM SWE README. The existing user edit expanded one gradient equation's delimiters; it was carried into renamed chapter 4 and applied consistently to all display equations. The new first chapters build the clamp task, executable checks, tool schemas, backend dispatch, state, and a scripted read/edit/test/submit trace before introducing token likelihoods.

All fifteen chapters use coding questions. Helpers are constructed locally with explanations; the checker rejects project imports and malformed display blocks. Three Mermaid diagrams explain the repair loop, runtime boundary, and sampled update cycle. The capstone trains a small patch-selection policy against actual test execution. Earlier domain source remains historical and is excluded from the site's pages, navigation, and downloads. Experiment 009 records verification and scope.
