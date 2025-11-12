import discord
from discord.ext import commands
from PIL import Image
import imagehash
import aiohttp
import io, json, os, re

# ---------------- CONFIG ---------------- #
TOKEN = os.environ.get("DISCORD_TOKEN")
HASH_DB_FILE = "hash_db.json"
GUILD_ID = None
os.makedirs("hash_db", exist_ok=True)

# ---------------- LOAD DB ---------------- #
if os.path.exists(HASH_DB_FILE):
    with open(HASH_DB_FILE, "r", encoding="utf-8") as f:
        hash_db = json.load(f)
else:
    hash_db = {}

# Temporary store for unnamed hashes
temp_hashes = []

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

def identify(img: Image.Image):
    ph = imagehash.phash(img)
    best_name, best_dist = None, 999
    for name, h in hash_db.items():
        d = ph - imagehash.hex_to_hash(h)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist

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

# ---------------- BOT SETUP ---------------- #
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    bot.session = aiohttp.ClientSession()
    print(f"Logged in as {bot.user}")

    # ---------- Identify Pokémon (context menu) ----------
    @bot.tree.context_menu(name="Identify Pokémon")
    async def identify_command(interaction: discord.Interaction, message: discord.Message):
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
            await interaction.response.send_message(f"Error reading image: {e}", ephemeral=True)
            return

        name, dist = identify(img)
        if name:
            name = real_name(name)
            await interaction.response.send_message(f"@Pokétwo#8236 c {name.lower()}", ephemeral=True)
        else:
            await interaction.response.send_message("❓ Could not identify the Pokémon.", ephemeral=True)

    # ---------- Store Pokémon Image Hash ----------
    @bot.tree.context_menu(name="Store Pokémon Image Hash")
    async def store_image_hash(interaction: discord.Interaction, message: discord.Message):
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
            ph = str(imagehash.phash(img))
        except Exception as e:
            await interaction.response.send_message(f"❌ Error reading image: {e}", ephemeral=True)
            return

        temp_hashes.append(ph)
        await interaction.response.send_message(f"⏳ Image hash stored temporarily ({len(temp_hashes)} stored)", ephemeral=True)

    # ---------- Assign Pokémon Name to Last Hash ----------
    @bot.tree.context_menu(name="Assign Pokémon Name to Last Hash")
    async def assign_name(interaction: discord.Interaction, message: discord.Message):
        if not temp_hashes:
            await interaction.response.send_message("⚠️ No stored image hashes to assign!", ephemeral=True)
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
            await interaction.response.send_message("❌ Could not find Pokémon name in this message!", ephemeral=True)
            return

        ph = temp_hashes.pop(0)
        hash_db[poke_name] = ph

        with open(HASH_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(hash_db, f, indent=2, ensure_ascii=False)

        await interaction.response.send_message(f"✅ Assigned name '{poke_name}' to stored hash", ephemeral=True)

    # ---------- Sync context menus ----------
    if GUILD_ID:
        await bot.tree.sync(guild=discord.Object(id=GUILD_ID))
    else:
        await bot.tree.sync()
    print("Context menus synced successfully!")

bot.run(TOKEN)

