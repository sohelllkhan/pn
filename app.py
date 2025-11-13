import discord
from discord.ext import commands
from PIL import Image
import aiohttp
import io, os, json, re
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.nn.functional import cosine_similarity

# ---------------- CONFIG ---------------- #
IMG_DB_DIR = "pokemon_images"
EMBED_DB_FILE = "pokemon_embeddings.json"
os.makedirs(IMG_DB_DIR, exist_ok=True)
GUILD_ID = None

# ---------------- HELPERS ---------------- #
def normalise_name(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.strip()
    s = re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', s)
    s = re.sub(r'[\.\,\!\?\:\;]+$', '', s)
    s = s.replace(' ', '_')
    return s
def real_name(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.strip()
    s = re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', s)
    s = re.sub(r'[\.\,\!\?\:\;]+$', '', s)
    s = s.replace('_', ' ')
    return s

def extract_pokemon_name_from_text(text: str):
    if not text:
        return None
    patterns = [
        re.compile(r'Wild\s+(.+?)\s+fled', re.IGNORECASE),
        re.compile(r'A wild\s+(.+?)\s+has fled', re.IGNORECASE),
        re.compile(r'The wild\s+(.+?)\s+fled', re.IGNORECASE)
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return normalise_name(m.group(1))
    return None

# ---------------- EMBEDDING MODEL ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights="DEFAULT").eval().to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor)
    return emb.squeeze(0).cpu()  # Return 1D tensor

# Load existing embeddings
if os.path.exists(EMBED_DB_FILE):
    with open(EMBED_DB_FILE, "r") as f:
        embed_db = json.load(f)
else:
    embed_db = {}  # { "pokemon_name": "filename.png" }

# ---------------- BOT SETUP ---------------- #
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
temp_images = []  # Temporarily store image bytes

@bot.event
async def on_ready():
    bot.session = aiohttp.ClientSession()
    print(f"Logged in as {bot.user}")

    # ---------- Store Pokémon Image ----------
    @bot.tree.context_menu(name="Store Pokémon Image")
    async def store_image(interaction: discord.Interaction, message: discord.Message):
        image_url = None
        if message.attachments:
            att = message.attachments[0]
            if att.content_type and att.content_type.startswith("image"):
                image_url = att.url
        elif message.embeds:
            embed = message.embeds[0]
            if embed.image and embed.image.url:
                image_url = embed.image.url

        if not image_url:
            await interaction.response.send_message("❌ No image found!", ephemeral=True)
            return

        async with bot.session.get(image_url) as resp:
            if resp.status != 200:
                await interaction.response.send_message("⚠️ Failed to fetch image!", ephemeral=True)
                return
            data = await resp.read()

        temp_images.append(data)
        await interaction.response.send_message(
            f"🕒 Image stored temporarily ({len(temp_images)} stored). Now assign its name.",
            ephemeral=True
        )

    # ---------- Assign Pokémon Name ----------
    @bot.tree.context_menu(name="Assign Pokémon Name")
    async def assign_name(interaction: discord.Interaction, message: discord.Message):
        if not temp_images:
            await interaction.response.send_message("⚠️ No stored images!", ephemeral=True)
            return

        poke_name = extract_pokemon_name_from_text(message.content)
        if not poke_name and message.embeds:
            for embed in message.embeds:
                if getattr(embed, "title", None):
                    poke_name = extract_pokemon_name_from_text(embed.title)
                if poke_name:
                    break
                if getattr(embed, "description", None):
                    poke_name = extract_pokemon_name_from_text(embed.description)
                if poke_name:
                    break

        if not poke_name:
            await interaction.response.send_message("❌ Could not find Pokémon name!", ephemeral=True)
            return

        data = temp_images.pop(0)
        save_path = os.path.join(IMG_DB_DIR, f"{poke_name}.png")
        with open(save_path, "wb") as f:
            f.write(data)

        # Calculate embedding and save to JSON
        img = Image.open(io.BytesIO(data)).convert("RGB")
        embedding = get_embedding(img).tolist()
        embed_db[poke_name] = embedding
        with open(EMBED_DB_FILE, "w") as f:
            json.dump(embed_db, f, indent=2)

        await interaction.response.send_message(f"✅ Saved `{poke_name}.png` with embedding.", ephemeral=True)

    # ---------- Identify Pokémon ----------
    @bot.tree.context_menu(name="Identify Pokémon")
    async def identify(interaction: discord.Interaction, message: discord.Message):
        image_url = None
        if message.attachments:
            att = message.attachments[0]
            if att.content_type and att.content_type.startswith("image"):
                image_url = att.url
        elif message.embeds:
            embed = message.embeds[0]
            if embed.image and embed.image.url:
                image_url = embed.image.url

        if not image_url:
            await interaction.response.send_message("❌ No image found!", ephemeral=True)
            return

        async with bot.session.get(image_url) as resp:
            if resp.status != 200:
                await interaction.response.send_message("⚠️ Failed to fetch image!", ephemeral=True)
                return
            data = await resp.read()

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            await interaction.response.send_message(f"❌ Error reading image: {e}", ephemeral=True)
            return

        new_emb = get_embedding(img)
        best_name, best_score = None, -1
        for name, emb_list in embed_db.items():
            emb_tensor = torch.tensor(emb_list)
            score = cosine_similarity(new_emb, emb_tensor, dim=0).item()
            if score > best_score:
                best_score, best_name = score, name

        if best_name and best_score > 0.95:  # threshold can be adjusted
            best_name = real_name(best_name)
            await interaction.response.send_message(f"@Pokétwo#8236 c {best_name.lower()}", ephemeral=True)
        else:
            await interaction.response.send_message("❓ Could not identify the Pokémon.", ephemeral=True)

    # ---------- Sync context menus ----------
    if GUILD_ID:
        await bot.tree.sync(guild=discord.Object(id=GUILD_ID))
    else:
        await bot.tree.sync()
    print(" Context menus synced!")

bot.run(os.environ.get("distok"))
