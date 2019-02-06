import click

from AttributeModelling.MeasureVAE.measure_vae import MeasureVAE
from AttributeModelling.MeasureVAE.vae_trainer import VAETrainer
from AttributeModelling.MeasureVAE.vae_tester import VAETester
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
@click.option('--num_epochs', default=30,
              help='number of training epochs')
@click.option('--train/--test', default=True,
              help='train or test the specified model')
@click.option('--plot/--no_plot', default=True,
              help='plot the training log')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--reg_loss/--no_reg_loss', default=True,
              help='train with regularization loss')
@click.option('--reg_type', default='rhy_complexity',
              help='attribute name string to be used for regularization')
@click.option('--reg_dim', default=0,
              help='dimension along with regularization is to be carried out')
@click.option('--attr_plot/--no_attr_plot', default=True,
              help='if True plots the attribute dsitributions, else produces interpolations')
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
         plot,
         log,
         reg_loss,
         reg_type,
         reg_dim,
         attr_plot
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
            lr=1e-4,
            has_reg_loss=reg_loss,
            reg_type=reg_type,
            reg_dim=reg_dim
        )
        trainer.train_model(
            batch_size=batch_size,
            num_epochs=num_epochs,
            plot=plot,
            log=log,
        )
    else:
        model.load()
        model.cuda()
        model.eval()

        tester = VAETester(
            dataset=folk_dataset_test,
            model=model,
            has_reg_loss=reg_loss,
            reg_type=reg_type,
            reg_dim=reg_dim
        )
        # tester.test_model(
        #    batch_size=batch_size
        # )
        # tester.test_interp()
        # tester.plot_transposition_points(plt_type='tsne')
        if attr_plot:
            grid_res = 0.05
            tester.plot_data_attr_dist(
                dim1=0,
                dim2=1,
            )
            tester.plot_attribute_surface(
                dim1=0,
                dim2=1,
                grid_res=grid_res
            )
            # tester.plot_attribute_surface(
            #    dim1=29,
            #    dim2=241,
            #    grid_res=grid_res
            # )
        else:
            tester.test_attr_reg_interpolations(
                dim=1,
            )


if __name__ == '__main__':
    main()
