import click

from AttributeModelling.MeasureVAE.measure_vae import MeasureVAE
from AttributeModelling.MeasureVAE.vae_trainer import VAETrainer
# from AttributeModelling.MeasureVAE.vae_tester import VAETester
from AttributeModelling.data.dataloaders.bar_dataset import *
from AttributeModelling.utils.helpers import *


@click.command()
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_encoder_layers', default=2,
              help='number of layers in encoder RNN')
@click.option('--encoder_hidden_size', default=512,
              help='hidden size of the encoder RNN')
@click.option('--encoder_dropout_prob', default=0.5,
              help='float, amount of dropout prob between encoder RNN layers')
@click.option('--has_metadata', default=False,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=256,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=512,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=20,
              help='number of training epochs')
@click.option('--train', is_flag=True, default='True',
              help='train or retrain the specified model')
def main(note_embedding_dim,
         metadata_embedding_dim,
         num_encoder_layers,
         encoder_hidden_size,
         encoder_dropout_prob,
         latent_space_dim,
         num_decoder_layers,
         decoder_hidden_size,
         decoder_dropout_prob,
         has_metadata,
         batch_size,
         num_epochs,
         train,
         ):

    is_short = False
    num_bars = 1
    folk_dataset_train = FolkNBarDataset(
        dataset_type='train',
        is_short=is_short,
        num_bars=num_bars)
    folk_dataset_test = FolkNBarDataset(
        dataset_type='test',
        is_short=is_short,
        num_bars=num_bars
    )

    model = MeasureVAE(
        dataset=folk_dataset_train,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_dropout_prob=encoder_dropout_prob,
        latent_space_dim=latent_space_dim,
        num_decoder_layers=num_decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_dropout_prob=decoder_dropout_prob,
        has_metadata=has_metadata
    )

    if train:
        if torch.cuda.is_available():
            model.cuda()
        trainer = VAETrainer(
            dataset=folk_dataset_train,
            model=model,
            lr=1e-3
        )
        trainer.train_model(
            batch_size=batch_size,
            num_epochs=num_epochs,
            plot=False,
            log=True,
        )
    else:
        model.load()
        model.cuda()
        model.eval()

    # tester = VAETester(
    #    dataset=folk_dataset_test,
    #    model=model
    # )
    # tester.test_model(eval_test=True)
    # tester.plot_transposition_points(plt_type='tsne')
    # tester.plot_attribute_dist(
    #    plt_type='tsne',
    #    attribute='beat_strength'
    # )


if __name__ == '__main__':
    main()
