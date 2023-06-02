import subprocess
import time
import os.path
import argparse

node_configs = {
    '1080Ti_dbg': [8],
    '1080Ti_special': [8],
    '1080Ti': [2],
    '1080Ti_short': [2],
    '1080Ti_slong': [2],
    '1080Ti_spec': [2],
    'P100': [5],
    'M40x8': [5],
    'M40x8_short': [5],
    'M40x8_slong': [5]
}


def get_commands(filename):
    lst = []
    with open(filename, 'r') as inf:
        lines = inf.readlines()
        lines = [x.strip() for x in lines if not x.strip().startswith('#')]
        for i in range(0, len(lines), 2):
            pair = (lines[i], lines[i+1])
            lst.append(pair)
    return lst


def get_current_jobs():
    process = subprocess.Popen(
        ['squeue', '-u', 'boyang'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    # print(out)
    lines = out.decode("ascii").strip().split('\n')
    # for id, l in enumerate(lines):
    #     print(str(id) + ' : ' + l + '\n')
    if len(lines) > 1:
        job_ids = [x.split()[0] for x in lines[1:]]
    else:
        job_ids = []
    return job_ids


def print_jobs_status(process_id):
    process = subprocess.Popen(
        ['squeue', '-u', 'boyang'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    # print(out)
    lines = out.decode("ascii").strip().split('\n')
    # for id, l in enumerate(lines):
    #     print(str(id) + ' : ' + l + '\n')
    if len(lines) > 1:
        print('  ' + lines[0] + '\n')
        for l in lines:
            if l.strip().startswith(process_id):
                print(l.strip() + '\n')


def save_batch_script(command):
    script_name = 'auto_script.sh'
    outf = open(script_name, 'w')
    outf.write('#!/bin/sh\n')
    #outf.write('module load python')
    outf.write(command + '\n')
    outf.close()
    return script_name


def submit_job(command, node_group, cpus):
    script_name = save_batch_script(command)

    if node_group in node_configs.keys():
        num_cpus = node_configs[node_group][0]
        command_list = ['sbatch', '-p', node_group, '--gres=gpu:1',
                        '--cpus-per-task', str(num_cpus), '-n', '1', script_name]
    elif node_group == 'KNL' or node_group == 'CPU':
        command_list = ['sbatch', '-p', 'KNL',
                        '--cpus-per-task', str(cpus), '-n', '1', script_name]
    else:
        raise ValueError('incorrect node specification: ' + str(node_group))

    print('command line : ' + ' '.join(command_list))
    job_proceess = subprocess.Popen(command_list, stdout=subprocess.PIPE)
    out, err = job_proceess.communicate()
    time.sleep(1)
    lines = out.decode("ascii").strip().split('\n')
    for l in lines:
        if l.startswith('Submitted batch job'):
            return l.split()[-1]


def check_terminated(process_id, process_dict):
    '''
    returns True if the process does not need to be resubmitted
    otherwise, returns False
    '''
    entry = process_dict[process_id]
    logfile = entry[1]
    slurm_log = './slurm-' + process_id + '.out'
    if os.path.isfile(logfile):
        with open(logfile, 'r') as inf:
            lines = inf.readlines()
        if 'TERMINATED' in lines[-1]:
            return True  # process terminated
    elif os.path.isfile(slurm_log):
        with open(slurm_log, 'r') as inf:
            lines = inf.readlines()
        text = ' '.join([x.strip() for x in lines])
        if 'Error' in text or 'Traceback' in text:
            # some error happened during the execution. Cannot recover. Must terminate
            print('process ' + process_id +
                  ' terminated with the following error')
            print(text)
            return True

    return False


def main(node_group, cpus=68, wakeup_interval=900, command_file='command_list.txt'):
    '''
    default values: 
        wake up interval = 15 minutes
        cpus, when running a cpu only task, is 68        
    '''
    command_list = get_commands(command_file)
    process_dict = {}

    for command, logfile in command_list:
        print('current jobs: ')
        jobs = get_current_jobs()
        print(', '.join(jobs))
        print('adding new job...')

        process_id = submit_job(command, node_group, cpus)
        print('acquired process id: ' + process_id + '\n')
        print_jobs_status(process_id)
        process_dict[process_id] = (command, logfile)

    while len(process_dict) > 0:
        time.sleep(wakeup_interval)
        jobs = get_current_jobs()
        print('current jobs ' + str(jobs))
        print('record: ' + str(process_dict))
        finished = [x for x in process_dict if x not in jobs]
        print('jobs that are finished: ' + str(finished))
        for pid in finished:
            if check_terminated(pid, process_dict):
                print('resubmitting: ' + command)
                process_dict.pop(pid)
                new_pid = submit_job(command, node_group, cpus)
                process_dict[new_pid] = (command, logfile)
            else:
                print('process ' + pid + ' terminated.')
                process_dict.pop(pid)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    # args = parser.parse_args()
    # print(args.accumulate(args.integers))
    parser = argparse.ArgumentParser(
        description='automatic submit jobs to slurm')
    parser.add_argument('-n', '--node', default='1080Ti',
                        help='the computational node group that the jobs are running on')
    parser.add_argument('-c', '--cpus', type=int, default=68,
                        help='the num of cpus needed for a job on CPU ONLY. Not needed for GPU jobs')
    parser.add_argument('-f', '--command_file', default='command_list.txt',
                        help='a file that contains the commands and log files')
    parser.add_argument('-i', '--interval', type=int, default=900,
                        help='the interval betweek waking up, in seconds')

    args = parser.parse_args()
    print('running tasks on ' + args.node +
          ', waking up every ' + str(args.interval) + ' seconds')
    main(args.node, cpus=args.cpus, wakeup_interval=args.interval,
         command_file=args.command_file)
