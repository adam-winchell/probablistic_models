import numpy as np
from sklearn.model_selection import KFold
import irt_functions as irt
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
import argparse
import pickle


def parse(arguments):
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    for key in arguments:
        type = arguments[key][0]
        help = arguments[key][1]
        parser.add_argument(key, type=type, help=help)

    args = parser.parse_args()
    return args

def pickle_in(filename):
    with open(filename, 'rb') as pickleFile:
        return pickle.load(pickleFile)

def get_kfold_random_state(x,y, num_folds, start_random_state):
    #finds a random state that will allow auc to be possible
    random_state = start_random_state

    while True:
        rskf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        result = True
        for train_index, test_index in rskf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                roc_auc_score(np.ravel(y_test), np.ravel(y_test))
            except:
                #couldn't calculate auc
                result = False
                break   #break for loop

        if result:
            print('Using random_state: ', random_state)
            return random_state
        else:
            random_state += 1







def run(x, y, model, question_type='', num_folds=10, num_draws=2500, debug=False):
    model_name = model+question_type
    print('Starting : '+model_name)

    random_state = get_kfold_random_state(x, y, num_folds, start_random_state=17)
    rskf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    fold_num = 1
    model_errors = []
    baseline_errors = []
    model_aucs = []
    baseline_aucs = []

    for train_index, test_index in rskf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        error, auc = getattr(irt, model)(x_train, x_test, y_train, y_test, num_draws=num_draws, fold_num=fold_num, model_name=model_name, debug=debug)

        #TODO consider adding fold weight back into calculations

        # if 'ppc' in model_name:
        #     # fold_weight = len(y_train) / len(y)
        #
        #     baseline_prediction = np.array(np.mean(y_train, axis=1)*y_train.shape[0])
        #     baseline_error = mean_squared_error(np.sum(baseline_prediction, axis=0), np.sum(y_train, axis=1))
        #     baseline_auc = roc_auc_score(y_train, baseline_prediction)
        #
        # else:
        #     # fold_weight = len(y_test) / len(y)
        #
        #     baseline_prediction = np.array([np.mean(y_train, axis=0)]*y_test.shape[0])
        #     baseline_error = mean_squared_error(np.sum(baseline_prediction, axis=1), np.sum(y_test, axis=1))
        #     baseline_auc = roc_auc_score(np.ravel(y_test), np.ravel(baseline_prediction))


        model_errors.append(error)
        model_aucs.append(auc)
        # baseline_errors.append(baseline_error)
        # baseline_aucs.append(baseline_auc)

        print('Fold %s, error=%s, auc=%s' % (fold_num, error, auc))
        # print('baseline, error=%s, auc=%s' % (baseline_error, baseline_auc))
        fold_num += 1
        if debug:
            quit()

    se_error = sem(model_errors)
    loss = np.mean(model_errors)/np.mean(baseline_errors)
    mean_auc = np.mean(model_aucs)
    se_auc = sem(model_aucs)

    print('************************************************************\n')
    print(model_name)
    print('model auc: ',mean_auc, ' , standard error: ', se_auc, '\n')
    print('baseline auc: ', np.mean(baseline_aucs), ' , standard error: ', sem(baseline_aucs), '\n')
    print('************************************************************\n')

    return mean_auc, se_auc, model_name




if __name__ == "__main__":
    x_dict = pickle_in('x_data_processed')
    y_dict = pickle_in('y_data_processed')

    arguments = {'model_name':[str, 'A string detailing the model name'], 'x_type':[str, 'A string detailing the type of x data'], 'y_type':[str, 'A string detailing the type of y data']}
    args = parse(arguments)
    model_name = args.model_name
    x_type = args.x_type
    y_type = args.y_type

    run(x_dict[y_type][x_type], y_dict[y_type], model_name, '_'+x_type+'_'+y_type)










