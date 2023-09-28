import numpy as np
import h5py
import time
import os


# functions (to be moved to utils.py)
def add_meta_keys(fn, pars_keys, image_keys=[]):
    with h5py.File(fn, 'r') as f:
        for key in f.keys():
            if key not in pars_keys and key not in image_keys:
                pars_keys.append(key)
    return 0


def get_square_images_fn(cdict, file_number=None):
    # in the future: load the first n_events from a file with more images
    event_type = cdict['event_type']
    n_events = cdict['n_events']
    mode = cdict['mode']
    Etrue_min = cdict['Etrue_min']
    fn = '%s_%i_images_%s' % (event_type, n_events, mode)
    if Etrue_min is not None and Etrue_min != 'None':
        fn += '_Etrue_min%.1fTeV' % Etrue_min
    if cdict.get('tel') != None:
        fn += '_%s' % cdict['tel']
    if file_number is not None:
        fn += '_file%i' % file_number
    fn += '.h5'
    return fn

def get_images_fns(cdict, folder=None, exists=False, nfiles=200):
    ev_types = cdict['model_events']
    #n_events = cdict['n_events']
    #n_events_tot = cdict.get('n_events_tot', None)
    #if n_events_tot == None:
    #    n_events_tot = n_events
    #nfiles = int(n_events_tot / n_events)
    out_dict = {}
    for k, event_type in enumerate(ev_types):
        cdict['event_type'] = event_type
        out_dict[event_type] = [get_square_images_fn(cdict, file_number=j+1) for j in range(nfiles)]
        if folder is not None and exists:
            out_dict[event_type] = [fn for fn in out_dict[event_type] if os.path.isfile(folder + fn)]
    return out_dict

def get_zeta_fns(cdict, folder=None, exists=False):
    out_dict = get_images_fns(cdict)
    for key in out_dict.keys():
        out_dict[key] = [fn.replace('.h5', '_zeta.h5') for fn in out_dict[key]]
        if folder is not None and exists:
            out_dict[key] = [fn for fn in out_dict[key] if os.path.isfile(folder + fn)]
    return out_dict


def load_images(folder, cdict):
    ev_types = cdict['model_events']
    n_events = cdict['n_events']
    n_events_tot = cdict.get('n_events_tot', None)
    if n_events_tot == None:
        n_events_tot = n_events
    nfiles = int(n_events_tot / n_events)
    data_key = cdict['data_key']
    print('load images')
    for k, event_type in enumerate(ev_types):
        print('load %s images' % event_type)
        cdict['event_type'] = event_type
        for j in range(nfiles):
            fn = folder + get_square_images_fn(cdict, file_number=j+1)
            with h5py.File(fn, 'r') as f:
                if k == 0 and j == 0:
                    dims = f[data_key].shape
                    out_dims = list(dims)
                    out_dims[0] = n_events_tot * len(ev_types)
                    images = np.zeros(out_dims, dtype=np.float32)
                ind_start = n_events_tot * k + dims[0] * j
                ind_end = n_events_tot * k + dims[0] * j + dims[0]
                fill_inds = list(range(ind_start, ind_end))
                images[fill_inds] = f[data_key][:]
    return images


def load_images_from_file(fn, key):
    with h5py.File(fn, 'r') as f:
        return f[key][:]


def get_group_key(key, f):
    for gkey in f.keys():
        if type(f[gkey]) != h5py._hl.dataset.Dataset and key in f[gkey].keys():
            return gkey
    return None

def load_meta_data(folder, cdict):
    ev_types = cdict['model_events']
    n_events = cdict['n_events']
    n_events_tot = cdict.get('n_events_tot', None)
    if n_events_tot == None:
        n_events_tot = n_events
    nfiles = int(n_events_tot / n_events)
    data_key = cdict['data_key']
    pars_keys = cdict['pars_keys']
    print('load meta data')
    for k, event_type in enumerate(ev_types):
        print(event_type)
        cdict['event_type'] = event_type
        for j in range(nfiles):
            fn = folder + get_square_images_fn(cdict, file_number=j+1)
            with h5py.File(fn, 'r') as f:
                if k == 0 and j == 0:
                    pars_dict = {}
                    for key in pars_keys:
                        gkey = get_group_key(key, f)
                        if gkey is None:
                            dims = [n_events]
                            out_dims = n_events_tot * len(ev_types)
                        else:
                            dims = f[gkey][key].shape
                            out_dims = list(dims)
                            out_dims[0] = n_events_tot * len(ev_types)
                        pars_dict[key] = np.zeros(out_dims, dtype=np.float32)
                ind_start = n_events_tot * k + dims[0] * j
                ind_end = n_events_tot * k + dims[0] * j + dims[0]
                fill_inds = list(range(ind_start, ind_end))
                for key in pars_dict.keys():
                    gkey = get_group_key(key, f)
                    if key == 'CR_type':
                        pars_dict[key][fill_inds] += int(event_type != 'proton')
                    elif gkey is not None:
                        pars_dict[key][fill_inds] = f[gkey][key][:]
                    else:
                        pass
    return pars_dict

def load_metadata_from_file(fn, key, event_type=None):
    with h5py.File(fn, 'r') as f:
        gkey = get_group_key(key, f)
        if key == 'CR_type' and event_type is not None:
            return int(event_type != 'proton')
        elif gkey is not None:
            return f[gkey][key][:]
        else:
            return None


# crop images
def get_min_max_inds(center, half_size, nmax):
    imin = np.ceil(center - half_size)
    imax = np.ceil(center + half_size)
    shift = -imin * np.heaviside(-imin, 0.) - (imax - nmax) * np.heaviside(imax - nmax, 0.)
    imin = (imin + shift).astype(int)
    imax = (imax + shift).astype(int)
    return imin, imax, shift
    

def crop_images(images, size, test=False, crop_fraction=0.03, boundary=5):
    t0 = time.time()
    di = 0.5 * size
    nn, nx, ny = images.shape
    res_arr = np.zeros((nn, size, size))
    norm = np.sum(images, axis=(1,2)) + 1.e-15
    t1 = time.time()
    
    # center of gravity along x
    ix = np.sum(np.sum(images, axis=2) * np.arange(nx), axis=1) / norm
    ix_min, ix_max, x_shift = get_min_max_inds(ix, di, nx)
    # center of gravity along y
    iy = np.sum(np.sum(images, axis=1) * np.arange(ny), axis=1) / norm
    iy_min, iy_max, y_shift = get_min_max_inds(iy, di, ny)
    
    t2 = time.time()
    
    # if True - the image is not cropped
    #crop_mask = np.abs(x_shift) + np.abs(y_shift) == 0.
    crop_mask = np.ones(nn, dtype=bool)
    t3 = time.time()
    
    
    for i in range(nn):
        res_arr[i] = images[i, ix_min[i]:ix_max[i], iy_min[i]:iy_max[i]]
        test_image = 1. * images[i]
        in_sum = np.sum(res_arr[i])
        test_image[ix_min[i]:ix_max[i], iy_min[i]:iy_max[i]] = 0.
        out_sum = np.sum(test_image)
        if in_sum == 0. or out_sum / in_sum > crop_fraction:
            crop_mask[i] = False

        test_image = 1. * images[i]
        b = boundary
        in_sum = np.sum(test_image[b:-b, b:-b])
        test_image[b:-b, b:-b] = 0.
        out_sum = np.sum(test_image)
        if in_sum == 0. or out_sum / in_sum > crop_fraction:
            crop_mask[i] = False

    t4 = time.time()
    if test:
        print('create arrays: %.3f s' % (t1 - t0))
        print('Get indices: %.3f s' % (t2 - t1))
        print('Crop mask: %.3f s' % (t3 - t2))
        print('Create final array: %.3f s' % (t4 - t3))

    return res_arr, crop_mask




def flatten_tel_images(images):
    '''
    flatten the number of telescopes dimension of the images
    '''
    ntot, image_size, image_size, ntel = images.shape
    im_new = np.zeros((ntel*ntot, image_size, image_size), dtype=np.float32)
    for i in range(ntel):
        im_new[i::ntel] = images[:,:,:,i]
    return im_new.reshape((ntel*ntot, image_size, image_size, 1))

def deflatten_tel_images(images, ntel):
    '''
    deflatten the number of telescopes dimension of the images
    '''
    ntot, image_size, image_size = images.shape[:3]
    ntot = int(ntot / ntel)
    im_new = np.zeros((ntot, image_size, image_size, ntel), dtype=np.float32)
    for i in range(ntel):
        im_new[:,:,:,i] = images[i::ntel]
    return im_new


def flatten_meta_data(data, ntel=4):
    if data.ndim == 1:
        data_loc = np.outer(data, np.ones(ntel))
    else:
        data_loc = 1. * data
    return data.flatten()
    
