# A Genetic Algorithm with PSO as Evolutionary Operator.
A Genetic Algorithm with PSO as Evolutionary Operator.


### How to set up the environment for the first time

Install `pip` and `virtualenv`:

`sudo apt-get install python3-pip`

`sudo pip3 install virtualenv`

Clone the project:

`git clone git@github.com:joseflauzino/ga_with_pso_as_evolutionary_operator.git`

Create a virtual environment within the project directory:

`python -m venv env`

Install the required dependencies:

`source env/bin/activate`

`pip install -r requirements.txt`


### How to run experiments

In the terminal run main and inform the function you want to test. 

`python main.py <function_name>`

Possible functions are: Sphere, Rastrigin, Ackley, Eggholder and drop_wave.
