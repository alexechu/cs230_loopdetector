import os
import numpy as np
from Bio.PDB import *
from scipy.spatial import distance_matrix

# From Raphael Eguchi
def getContactMap(pdb):
    structure = PDBParser(QUIET=True).get_structure(pdb[:-3], pdb)
    A = []
    for model in structure:
        for chain in model:
            for res in chain:
                try:
                    A.append(np.asarray(res['CA'].get_coord())) # C-alpha coordinates are extracted in residue order.
                except:
                    continue
    return distance_matrix(A,A)

def get_dssp_string(pdb):
    try:
        structure = PDBParser(QUIET=True).get_structure(pdb[:-3], pdb)
        model = structure[0]
        dssp = DSSP(model, pdb, dssp='mkdssp')
        ss_assignments = [dssp[k][2] for k in dssp.keys()]    # list of strings 'E' '-' etc indicating SS for each residue
        dssp_string = ''.join(ss_assignments)
        return dssp_string
    except:
        raise Exception('Error in get_dssp_string()...')

def loop_labels_from_dssp_string(dssp):
    is_loop = ['T', 'S', '-']
    labels = [ss in is_loop for ss in dssp]
    return np.array(labels)

def y_from_loop_labels(raw):
    # TODO 
    raw = raw.astype(int)

w = 64
s = 10
x_list = []
y_list = []
count = 0
for file in os.listdir('.'):
    if file.endswith('.pdb'):
        try:
            contact_map = getContactMap(file)
            contact_map = contact_map.reshape((contact_map.shape[0], contact_map.shape[1], 1))
            loop_labels = loop_labels_from_dssp_string(get_dssp_string(file))
            loop_labels = loop_labels.reshape((len(loop_labels), 1, 1))
            if contact_map.shape[0] == loop_labels.shape[0]:
                # Sliding windows
                x = 0
                y = 0
                while (x+w) <= contact_map.shape[0]:
                    while (y+w) <= contact_map.shape[1]:
                        window = contact_map[x:x+w,y:y+w,:]
                        labels = loop_labels[x:x+w,:,:]
                        x_list.append(window)
                        y_list.append(labels)
                        x += s
                        y += s
            if count % 1000 == 0:
                print('Processed ' + str(count) + ' structures...')
            count += 1
        except:
            continue
    else:
        continue

print('Processed ' + str(count) + ' structures...')
X = np.concatenate(x_list, axis=2)
Y = np.concatenate(y_list, axis=2)

if X.shape[0] == Y.shape[0] and X.shape[2] == Y.shape[2]:
    # Split train, dev, test
    m = X.shape[2]
    print('Number of examples: ' + str(m))
    train = int(m*0.8)
    dev = int(m*0.9)
    idxs = np.random.permutation(m)
    train_idxs, dev_idxs, test_idxs = idxs[:train], idxs[train:dev], idxs[dev:]
    train_X = X[:,:,train_idxs]
    train_Y = Y[:,:,train_idxs]
    dev_X = X[:,:,dev_idxs]
    dev_Y = Y[:,:,dev_idxs]
    test_X = X[:,:,test_idxs]
    test_Y = Y[:,:,test_idxs]

    # Save
    print(X.shape)
    print(Y.shape)
    print(train_X.shape)
    print(train_Y.shape)
    print(dev_X.shape)
    print(dev_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    np.save('all_X.npy', X)
    np.save('all_Y.npy', Y)
    np.save('train_X.npy', train_X)
    np.save('train_Y.npy', train_Y)
    np.save('dev_X.npy', dev_X)
    np.save('dev_Y.npy', dev_Y)
    np.save('test_X.npy', test_X)
    np.save('test_Y.npy', test_Y)

