import os
import shutil

import task
import neat
import hpneat
import visualize

import hpneat_config

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

def eval_fitness_A(net,print_=False):
    """
    Evaluates fitness of the genome that was used to generate 
    provided net
    Arguments:
        net: The feed-forward neural network generated from genome
    Returns:
        The fitness score - the higher score the means the better 
        fit organism. Maximal score: 16.0
    """
    t = task.task(5, 2)
    history = ''
    mymodel = hpneat.ModulatoryHebbianModel(net, hpneat_config)
    mymodel.fitness = 0

    for step in range(20):

        t.step = step

        output = mymodel.activate([1,0,0,0])
        if(print_):
            print('step', step, 'output:', output)
        if (t.rule == output.index(max(output)) ):
            mymodel.fitness += 1
            mymodel.activate([0,1,1,0])
            history += 'o'
        else:
            mymodel.activate([0,1,0,1])
            history += 'x'
    if(print_):
        print(history)

    return mymodel.fitness

def eval_fitness_B(net,print_=False):
    t4 = task.task(4, 2)
    t5 = task.task(5, 2)
    t6 = task.task(6, 2)

    history4 = ''
    history5 = ''
    history6 = ''

    mymodel4 = hpneat.SeparatedModulatoryModel(net, hpneat_config)
    mymodel5 = hpneat.SeparatedModulatoryModel(net, hpneat_config)
    mymodel6 = hpneat.SeparatedModulatoryModel(net, hpneat_config)

    mymodel4.fitness = 0
    mymodel5.fitness = 0
    mymodel6.fitness = 0

    for step in range(20):

        t4.step = step
        t5.step = step
        t6.step = step

        output4 = mymodel4.activate([1,0,0,0])
        output5 = mymodel5.activate([1,0,0,0])
        output6 = mymodel6.activate([1,0,0,0])
        if(print_):
            print('step', step, 'output4:', output4)
            print('step', step, 'output5:', output5)
            print('step', step, 'output6:', output6)

        if (t4.rule == output4.index(max(output4)) ):
            mymodel4.fitness += 1
            mymodel4.activate([0,1,1,0])
            history4 += 'o'
            if(t4.is_bonus == True): #ドンピシャでボーナス
                mymodel4.fitness += 1
        else:
            mymodel4.activate([0,1,0,1])
            history4 += 'x'

        if (t5.rule == output5.index(max(output5)) ):
            mymodel5.fitness += 1
            mymodel5.activate([0,1,1,0])
            history5 += 'o'
            if(t5.is_bonus == True): #ドンピシャでボーナス
                mymodel5.fitness += 1
        else:
            mymodel5.activate([0,1,0,1])
            history5 += 'x'

        if (t6.rule == output6.index(max(output6)) ):
            mymodel6.fitness += 1
            mymodel6.activate([0,1,1,0])
            history6 += 'o'
            if(t6.is_bonus == True): #ドンピシャでボーナス
                mymodel6.fitness += 1
        else:
            mymodel6.activate([0,1,0,1])
            history6 += 'x'

    if(print_):
        print(history4)
        print(history5)
        print(history6)

    return (mymodel4.fitness + mymodel5.fitness + mymodel6.fitness) /3

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    XOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #genome.fitness = eval_fitness_A(net)
        genome.fitness = eval_fitness_B(net)

def run_experiment(config_file):
    """
    The function to run XOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/neat-checkpoint-'))

    # Run for up to 1000 generations.
    best_genome = p.run(eval_genomes,1000)

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Show output of the most fit genome against training data.
    """
    print('\nOutput:')
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    """

    # Check if the best genome is an adequate XOR solver
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    #best_genome_fitness = eval_fitness_A(net,True)
    best_genome_fitness = eval_fitness_B(net,True)
    if best_genome_fitness >= config.fitness_threshold:
        print("\n\nSUCCESS: The solver found!!!")
    else:
        print("\n\nFAILURE: Failed to find solver!!!")

    # Visualize the experiment results
    node_names = {-1:'1', -2:'2', -3:'3', -4:'4', -5:'5', -6:'6', -7:'7', -8:'8', 0:'0', 1:'1', 2:'2', 3:'3'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'neat_config.ini')

    # Clean results of previous run if any or init the ouput directory
    clean_output()

    # Run the experiment
    run_experiment(config_path)
