## Sokoban game using BFS and A* search
With A*, using two heuristic funtions:\
- Simple heuristic: (using manhattan distance)*number of target left\
- Improved heuristic: (find the smallest distance from each position to its nearest target)*number of box moved*target left\
## Run
python .\main.py [level] [algorithm] [simulate sol] [file]\
- [level]: type value from 1 to 15 (levels in input file)\
- [algorithm]: - type 'a' for a* search with simple heuristic func\
               - type 'a2' for a* search with improved heuristic func\
               - type 'b' for BFS\
- [simulate sol]: type '-s' if u want to animate the solution (default off)\
- [file]: type '-f filename' for input level (default 'levels.txt')\
- Ex:  python .\main.py 1 a (without animate sol)\
      python .\main.py 1 a -s (with animate sol)
