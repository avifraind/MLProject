import numpy
import numpy as np
import torch
from dataset import PhotoMonetDataset
import matplotlib.pyplot as plt
import sys
from utils import save_checkpoint, load_checkpoint, LambdaLR
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(disc_M, disc_P, gen_P, gen_M, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    M_reals = 0
    M_fakes = 0
    loop = tqdm(loader, leave=True)

    ret_D_loss = [0]
    ret_loss_G_P = [0]
    ret_loss_G_H = [0]
    ret_cycle_photo_loss = [0]
    ret_cycle_monet_loss = [0]
    ret_identity_monet_loss = [0]
    ret_identity_photo_loss = [0]
    ret_G_loss = [0]
    for idx, (photo, monet) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        monet = monet.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_monet = gen_M(photo)
            D_M_real = disc_M(monet)
            D_M_fake = disc_M(fake_monet.detach())
            M_reals += D_M_real.mean().item()
            M_fakes += D_M_fake.mean().item()
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            fake_photo = gen_P(monet)
            D_P_real = disc_P(photo)
            D_P_fake = disc_P(fake_photo.detach())
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            # put it togethor
            D_loss = (D_M_loss + D_P_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_M(fake_monet)
            D_P_fake = disc_P(fake_photo)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

            # cycle loss
            cycle_photo = gen_P(fake_monet)
            cycle_monet = gen_M(fake_photo)
            cycle_photo_loss = l1(photo, cycle_photo)
            cycle_monet_loss = l1(monet, cycle_monet)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_photo = gen_P(photo)
            identity_monet = gen_M(monet)
            identity_photo_loss = l1(photo, identity_photo)
            identity_monet_loss = l1(monet, identity_monet)

            # add all togethor
            G_loss = (
                    loss_G_P
                    + loss_G_H
                    + cycle_photo_loss * config.LAMBDA_CYCLE
                    + cycle_monet_loss * config.LAMBDA_CYCLE
                    + identity_monet_loss * config.LAMBDA_IDENTITY
                    + identity_photo_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        ret_D_loss[0]=float(D_loss)
        ret_loss_G_P[0]=float(loss_G_P)
        ret_loss_G_H[0]= float(loss_G_H)
        ret_cycle_photo_loss[0]= float(cycle_photo_loss)
        ret_cycle_monet_loss[0]=float(cycle_monet_loss)
        ret_identity_monet_loss[0]= float(identity_monet_loss)
        ret_identity_photo_loss[0]=float(identity_photo_loss)
        ret_G_loss[0]=float(G_loss)
        if idx % 100 == 0:


            save_image(fake_monet * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/fakehorse/fakehorse_{idx}.png")
            save_image(fake_photo * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/fakezebra/fakezebra_{idx}.png")
            save_image(monet * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/horse/horse_{idx}.png")
            save_image(photo * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/zebra/zebra_{idx}.png")

        loop.set_postfix(M_real=M_reals / (idx + 1), M_fake=M_fakes / (idx + 1))

    return {"D_loss": ret_D_loss, "loss_G_P": ret_loss_G_P,
            "loss_G_H": ret_loss_G_H,
            "cycle_photo_loss": ret_cycle_photo_loss,
            "cycle_monet_loss": ret_cycle_monet_loss,
            "identity_monet_loss": ret_identity_monet_loss,
            "identity_photo_loss": ret_identity_photo_loss, "G_loss": ret_G_loss}


def val_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()

        save_image(fake_horse * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/val/fakehorse/fakehorse_{idx}.png")
        save_image(fake_zebra * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/val/fakezebra/fakezebra_{idx}.png")
        save_image(horse * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/val/horse/horse_{idx}.png")
        save_image(zebra * 0.5 + 0.5, config.BASE_DIR + f"/data/saved_images/val/zebra/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    import gc

    gc.collect()
    # zebra==photo
    # horse==monet
    torch.cuda.empty_cache()

    disc_M = Discriminator(in_channels=3).to(config.DEVICE)
    disc_P = Discriminator(in_channels=3).to(config.DEVICE)
    gen_P = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_M = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_P.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_P.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=LambdaLR(config.NUM_EPOCHS, 0,
                                                                                   config.NUM_EPOCHS / 2).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=LambdaLR(config.NUM_EPOCHS, 0,
                                                                                    config.NUM_EPOCHS / 2).step)
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_M, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_P, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_M, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_P, opt_disc, config.LEARNING_RATE,
        )

    dataset = PhotoMonetDataset(
        root_monet=config.TRAIN_DIR + "/monet_jpg30", root_photo=config.TRAIN_DIR + "/photo_jpg",
        transform=config.transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    losses = {"D_loss": [], "loss_G_P": [],
              "loss_G_H": [],
              "cycle_photo_loss": [],
              "cycle_monet_loss": [],
              "identity_monet_loss": [],
              "identity_photo_loss": [],
              "G_loss": []}
    for epoch in range(config.NUM_EPOCHS):
        print("-" * 4 + "Epoch number: " + str(epoch) + "-" * 4)
        d = train_fn(disc_M, disc_P, gen_P, gen_M, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        for key, val in d.items():
            print("key: " + key + ". " + "val: " + str(val))
            losses[key] = losses[key] + val
            plt.title(key)
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.plot( np.array(range(epoch+1)),np.array(losses[key]))
            plt.show()

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if config.SAVE_MODEL:
            save_checkpoint(gen_M, opt_gen,
                            filename=str(config.LEARNING_RATE) + "_" + str(config.LAMBDA_IDENTITY) + "_" + str(
                                config.LAMBDA_CYCLE) + config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_P, opt_gen,
                            filename=str(config.LEARNING_RATE) + "_" + str(config.LAMBDA_IDENTITY) + "_" + str(
                                config.LAMBDA_CYCLE) + config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_M, opt_disc,
                            filename=str(config.LEARNING_RATE) + "_" + str(config.LAMBDA_IDENTITY) + "_" + str(
                                config.LAMBDA_CYCLE) + config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_P, opt_disc,
                            filename=str(config.LEARNING_RATE) + "_" + str(config.LAMBDA_IDENTITY) + "_" + str(
                                config.LAMBDA_CYCLE) + config.CHECKPOINT_CRITIC_Z)

    val_dataset = PhotoMonetDataset(
        root_monet=config.VAL_DIR + "/monet_jpg", root_photo=config.VAL_DIR + "/photo_jpg",
        transform=config.transforms_val
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    val_fn(disc_M, disc_P, gen_P, gen_M, val_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)


if __name__ == "__main__":

    main()
