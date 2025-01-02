installazione pulita nella cartella plansys2_project

rm -rf build/ install/ log/

colcon build --symlink-install

source install/local_setup.bash

esecuzione ros2

ros2 launch plansys2_project plansys2_project_launch.py


in un nuovo terminale 

ros2 run plansys2_terminal plansys2_terminal

source launch/commands

run plan-file launch/plan.txt








pddl4j

java -cp build/libs/pddl4j-4.0.0.jar fr.uga.pddl4j.planners.statespace.FF \
   src/pddl/domain.pddl \
   src/pddl/problemTask1.pddl
