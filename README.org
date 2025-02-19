~some basic info about the project here~

*  Repository skeleton
** Key components

| name           | description                                                       |
|----------------+-------------------------------------------------------------------|
| cfg            | Config files                                                      |
| explorations   | All exploratory analysis                                          |
| reports        | Most textual outputs (papers, posters, etc)                       |
| src            | All src codes (core routines, data pre-processing)                |
| visualizations | All codes that generate the figures of papers, poster, talks, etc |

** Directory structure 
#+BEGIN_SRC sh :results output :exports results :cache yes
tree -F -d -L 2 ./
#+END_SRC

#+RESULTS[148c8ceb23035813c239f10b432368b6c6ab16b5]:
#+begin_example
./
├── bku
├── cfg
├── explorations
│   ├── notebooks
│   ├── soa
│   └── unpublished
├── ext
├── localdata
├── notes
├── reports
│   ├── AbstractsPosters
│   ├── confs
│   ├── misc
│   ├── papers
│   ├── presentations
│   ├── proposals
│   ├── sot
│   └── sreps
├── sandbox
├── shared
│   └── lnJrnlYYYY_shortTitleXXXXX
├── src
│   ├── data
│   ├── exptools
│   ├── methods
│   ├── misc
│   ├── packages
│   ├── simulations
│   └── utilities
├── tmp
├── tutorials
├── visualizations
│   ├── AbstractsPosters
│   ├── confPapers
│   ├── misc
│   ├── papers
│   └── presentations
└── void

38 directories
#+end_example


#+BEGIN_SRC sh :results output :exports results :cache yes
tree -F -d -L 2 ./
#+END_SRC

#+RESULTS[148c8ceb23035813c239f10b432368b6c6ab16b5]:
#+begin_example
./
├── bku
├── cfg
├── explorations
│   ├── notebooks
│   ├── soa
│   └── unpublished
├── ext
├── localdata
├── notes
├── reports
│   ├── AbstractsPosters
│   ├── papers
│   ├── proposals
│   ├── sot
│   └── sreps
├── sandbox
├── src
│   ├── data
│   ├── exptools
│   ├── methods
│   ├── misc
│   ├── packages
│   ├── simulations
│   └── utilities
├── tmp
├── tutorials
├── visualizations
│   ├── AbstractsPosters
│   ├── confPapers
│   ├── misc
│   ├── papers
│   └── presentations
└── void

33 directories
#+end_example
