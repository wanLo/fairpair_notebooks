from deepface.DeepFace import *
import scipy.io


def analyze_quicker(
    images,
    actions=("emotion", "age", "gender", "race"),
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    silent=False,
):

    # ---------------------------------
    # validate actions
    if isinstance(actions, str):
        actions = (actions,)

    actions = list(actions)
    # ---------------------------------
    # build models
    models = {}
    #if "emotion" in actions:
    #    models["emotion"] = build_model("Emotion")

    if "age" in actions:
        models["age"] = build_model("Age")

    if "gender" in actions:
        models["gender"] = build_model("Gender")

    if "race" in actions:
        models["race"] = build_model("Race")
    # ---------------------------------

    image_df = pd.DataFrame()

    for img_path in images:
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=(224, 224),
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
        tmp_df = pd.DataFrame.from_records(img_objs, columns=['img_content', 'img_region', 'confidence'])
        tmp_df['img_path'] = img_path

        image_df = pd.concat([image_df, tmp_df]).reset_index(drop=True)

    batch_iter = 0
    batch_step = 100

    results_df = pd.DataFrame()
    
    while batch_iter < len(image_df):

        batch_df = image_df.iloc[batch_iter:batch_iter+batch_step]

        if len(batch_df) > 0:
            tmp_results_df = batch_df[['img_path', 'img_region', 'confidence']].reset_index(drop=True)
            batch_content = np.stack(batch_df.img_content)[:,0,:,:]  # results in batch size * width * height * RGB

            gender_prediction = models["gender"].predict(batch_content, verbose=0)
            gender_df = pd.DataFrame(gender_prediction, columns=Gender.labels)

            age_prediction = models["age"].predict(batch_content, verbose=0)
            age_df = pd.DataFrame(age_prediction, columns=np.arange(0,101))
            
            race_prediction = models["race"].predict(batch_content, verbose=0)
            race_df = pd.DataFrame(race_prediction, columns=Race.labels)

            tmp_results_df = pd.concat([tmp_results_df, gender_df, race_df, age_df], axis=1)

            results_df = pd.concat([results_df, tmp_results_df])
        batch_iter += batch_step
    
    return results_df.reset_index(drop=True)


if __name__ == "__main__":

    wiki = scipy.io.loadmat('./data/imdb-wiki/wiki.mat')
    imdb = scipy.io.loadmat('./data/imdb-wiki/imdb.mat')

    full_path_wiki = np.stack(wiki['wiki']['full_path'][0][0][0]).flatten()
    full_path_imdb = np.stack(imdb['imdb']['full_path'][0][0][0]).flatten()

    wiki_df = pd.DataFrame(full_path_wiki, columns=['img_path'])
    wiki_df['img_path'] = './data/wiki_crop/' + wiki_df['img_path']
    imdb_df = pd.DataFrame(full_path_imdb, columns=['img_path'])
    imdb_df['img_path'] = './data/imdb_crop/' + imdb_df['img_path']
    pictures_df = pd.concat([wiki_df, imdb_df]).reset_index(drop=True)

    #print(pictures_df.img_path.head(5))

    picture_batch_iter = 0
    picture_batch_step = 100

    annotated_df = pd.DataFrame()

    while picture_batch_iter < len(pictures_df):
        print(f'processing pictures {picture_batch_iter} to {picture_batch_iter + picture_batch_step}')

        annotated_tmp_df = analyze_quicker(images=list(pictures_df.iloc[picture_batch_iter:picture_batch_iter+picture_batch_step].img_path),
                                           enforce_detection=False)
        
        annotated_tmp_df.to_csv(f'./data/annotated_results/annotated_pictures_{picture_batch_iter}.csv', index=False)
        annotated_df = pd.concat([annotated_df, annotated_tmp_df]).reset_index(drop=True)

        picture_batch_iter += picture_batch_step