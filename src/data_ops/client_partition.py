import os
import random
import numpy as np
import tensorflow as tf

from src.data_ops.attacks import apply_attacks_for_client, assign_fake_clients

def load_image(path, img_size=(224, 224)):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3,expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    return img

def create_client_dataset(
    image_paths,
    labels,
    client_id,
    malicious_clients_attack_map,
    attack_params,
    img_size=(224, 224),
    model_size=(64, 64),
    seed=42,
    batch_size=8,
    test_split=0.2
):

    images = []
    loaded_labels = []

    for i, path in enumerate(image_paths):
        img = load_image(path, img_size)
        img = tf.image.resize(img, model_size)
        images.append(img)
        loaded_labels.append(labels[i])

    labels = np.array(loaded_labels, dtype=np.int32)
    n = len(images)

    if n == 0:
        empty_images = tf.zeros((0, model_size[0], model_size[1],3),dtype=tf.float32)
        empty_labels = tf.zeros((0,), dtype=tf.int32)
        empty_ds = tf.data.Dataset.from_tensor_slices((empty_images, empty_labels)).batch(batch_size)
        return empty_ds, empty_ds

    indices = np.arange(n)
    split_rng = (np.random.RandomState(seed))
    split_rng.shuffle(indices)
    raw_test_size = int(np.floor(n * float(test_split)))

    if n > 1 and test_split > 0:
        test_size = max(1, raw_test_size)
        test_size = min(test_size, n - 1)
    else:
        test_size = 0

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    images_arr = np.stack([img.numpy() if tf.is_tensor(img) else img for img in images]).astype(np.float32)

    x_train = images_arr[train_idx]
    y_train = labels[train_idx]
    x_test = images_arr[test_idx]
    y_test = labels[test_idx]
    train_images_list = [img for img in x_train]

    (attacked_train_images, attacked_train_labels) = apply_attacks_for_client(images=train_images_list,
                                                                              labels=np.array(y_train,dtype=np.int32),
                                                                              client_id=client_id,
                                                                              malicious_clients_attack_map=malicious_clients_attack_map,
                                                                              attack_params=attack_params,
                                                                              seed=seed)

    x_train = np.stack([img.numpy() if tf.is_tensor(img) else img for img in attacked_train_images]).astype(np.float32)
    y_train = np.array(attacked_train_labels, dtype=np.int32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    if len(x_train) > 0:
        train_ds = train_ds.shuffle(buffer_size=len(x_train), seed=seed)

    train_ds = train_ds.batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)

    print(f"{client_id}: total={n}, train={len(x_train)}, test={len(x_test)}, attack={malicious_clients_attack_map.get(client_id)}")

    return train_ds, test_ds

def prepare_client_dataset(
    data_dir,
    num_client,
    num_malicious,
    attack_list,
    attack_params,
    seed=42,
    max_samples=10000,
    attack_distribution=None,
    test_split=0.2,
    batch_size=8
):

    image_paths = []
    labels = []
    class_names = sorted([name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
        ])

    class_labels = {name: idx for idx, name in enumerate(class_names)}
    print("class_labels:",class_labels)

    for class_name in class_names:
        class_path = os.path.join(data_dir,class_name)
        for file_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, file_name))
            labels.append(class_labels[class_name])

    combined = list(zip(image_paths,labels))
    rng = random.Random(seed)
    rng.shuffle(combined)

    if max_samples is not None:
        combined = combined[:max_samples]

    image_paths, labels = zip(*combined)

    (malicious_clients,malicious_clients_attack_map,all_client_ids) = assign_fake_clients(num_client= num_client,
                                                                                          num_malicious= num_malicious,
                                                                                          attack_list= attack_list,
                                                                                          seed=seed,
                                                                                          exclude_head=True,
                                                                                          additional_malicious_clients=[],
                                                                                          max_fake_ratio=0.7,
                                                                                          attack_distribution=attack_distribution
                                                                                          )

    print("malicious_clients:",malicious_clients)
    print("attack map:",malicious_clients_attack_map)
    clients_datasets = {}
    client_test_datasets = {}
    total_images = len(image_paths)

    for i, client_id in enumerate(all_client_ids):

        start = (i * total_images // num_client)
        end = ((i + 1) * total_images // num_client)

        train_ds, test_ds = create_client_dataset(image_paths= image_paths[start:end],
                                                   labels=labels[start:end],
                                                   client_id=client_id,
                                                   malicious_clients_attack_map=malicious_clients_attack_map,
                                                   attack_params=attack_params,
                                                   seed=seed + i,
                                                   batch_size=batch_size,
                                                   test_split=test_split)

        clients_datasets[client_id] = train_ds
        client_test_datasets[client_id] = test_ds

    return clients_datasets, client_test_datasets, malicious_clients, malicious_clients_attack_map